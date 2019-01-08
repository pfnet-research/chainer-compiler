#!/usr/bin/python3

import argparse
import json
import os
import sys

import topi
import tvm
from tvm import autotvm


class TVMConfig(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.cfgs = []
            for lineno, line in enumerate(f):
                if line.startswith('#'):
                    continue
                self.cfgs.append((lineno + 1, json.loads(line)))

    def show_entries(self, query):
        for lineno, cfg in self.cfgs:
            if self.matches(cfg, query):
                print(lineno, json.dumps(cfg))

    def contains(self, query):
        for lineno, cfg in self.cfgs:
            if self.matches(cfg, query):
                return True
        return False

    def matches(self, cfg, query):
        target, op, args, kwargs, info, tvm_params = cfg['i']
        results = cfg['r']
        if len(info) != len(query):
            return False
        for q, i in zip(query, info):
            if q is not None and q != i:
                return False
        return True


def read_tasks(op_filenames, batchsize):
    tasks = {}
    for op_filename in op_filenames:
        with open(op_filename) as f:
            task = json.load(f)
            if batchsize:
                task['bsize'] = batchsize
            task_key = tuple(sorted(task.items()))
            if task_key not in tasks:
                tasks[task_key] = (op_filename, task)
            else:
                print('%s has dup parameter set' % op_filename)
    return tasks


def autotvm_key_from_task(task):
    assert task['op'] in ('Conv', 'ConvTranspose')
    bsize = task['bsize']
    ichan = task['ichan']
    ochan = task['ochan']
    kernel_h = task['kernel_h']
    kernel_w = task['kernel_w']
    height = task['height']
    width = task['width']
    stride_h = task['stride_h']
    stride_w = task['stride_w']
    pad_h = task['pad_h']
    pad_w = task['pad_w']
    dtype = task['dtype'].lower()

    workload_op = {
        'Conv': 'conv2d',
        'ConvTranspose': 'conv2d_transpose_nchw',
    }[task['op']]
    key = [workload_op]
    key.append([bsize, ichan, height, width, dtype])
    key.append([ochan, ichan, kernel_h, kernel_w, dtype])
    key.append([stride_h, stride_w])
    key.append([pad_h, pad_w])
    key.append([1, 1])
    key.append('NCHW')
    key.append(dtype)
    return key

def deserialize_args(args):
    ret = []
    for t in args:
        if isinstance(t, tuple) and t[0] == 'TENSOR':
            ret.append(tvm.placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret


@autotvm.task.register('topi_nn_conv2d')
def topi_nn_conv2d(*args, **kwargs):
    input, weight, *args = deserialize_args(args)
    c = topi.nn.conv2d(input, weight, *args, **kwargs)
    s = topi.generic.schedule_conv2d_nchw([c])
    return s, [input, weight, c]


@autotvm.task.register('topi_nn_conv2d_transpose_nchw')
def topi_nn_conv2d_transpose_nchw(*args, **kwargs):
    input, weight, *args = deserialize_args(args)
    c = topi.nn.conv2d_transpose_nchw(input, weight, *args, **kwargs)
    s = topi.generic.schedule_conv2d_transpose_nchw([c])
    return s, [input, weight, c]


def autotvm_task(task, target):
    assert task['op'] in ('Conv', 'ConvTranspose')
    bsize = task['bsize']
    ichan = task['ichan']
    ochan = task['ochan']
    kernel_h = task['kernel_h']
    kernel_w = task['kernel_w']
    height = task['height']
    width = task['width']
    stride_h = task['stride_h']
    stride_w = task['stride_w']
    pad_h = task['pad_h']
    pad_w = task['pad_w']
    dtype = task['dtype'].lower()

    if task['op'] == 'Conv':
        args = (
            ('TENSOR', (bsize, ichan, height, width), dtype),
            ('TENSOR', (ochan, ichan, kernel_h, kernel_w), dtype),
            (stride_h, stride_w),
            (pad_h, pad_w),
            (1, 1),
            'NCHW',
            dtype
        )
        return autotvm.task.create(topi_nn_conv2d,
                                   args,
                                   target,
                                   template_key='direct')
    elif task['op'] == 'ConvTranspose':
        args = (
            ('TENSOR', (bsize, ichan, height, width), dtype),
            ('TENSOR', (ochan, ichan, kernel_h, kernel_w), dtype),
            (stride_h, stride_w),
            (pad_h, pad_w),
            dtype
        )
        return autotvm.task.create(topi_nn_conv2d_transpose_nchw,
                                   args,
                                   target,
                                   template_key='direct')


# From https://github.com/dmlc/tvm/blob/master/tutorials/autotvm/tune_nnvm_cuda.py
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def main():
    parser = argparse.ArgumentParser(description='Tune ops')
    parser.add_argument('output', type=str)
    parser.add_argument('ops', type=str, nargs='+')
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--base', type=str)
    parser.add_argument('--target', type=str, default='cuda')
    args = parser.parse_args()

    tasks = read_tasks(args.ops, args.batchsize)

    print('Read %d tasks from %d files' % (len(tasks), len(args.ops)))

    if args.base:
        base_config = TVMConfig(args.base)
        discard_keys = []
        for task_key, (filename, task) in tasks.items():
            query = autotvm_key_from_task(task)
            if base_config.contains(query):
                print('%s is already tuned' % filename)
                discard_keys.append(task_key)
        for task_key in discard_keys:
            tasks.pop(task_key)
        print('Removed %d tasks. Will tune for %d tasks.' %
              (len(discard_keys), len(tasks)))

    tuning_opt = {
        'log_filename': args.output,

        'tuner': 'xgb',
        'n_trial': 2000,
        'early_stopping': 600,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4),
        ),
    }

    tvm_tasks = []
    for task_key, (filename, task) in tasks.items():
        print('Tuning for %s' % filename)
        tvm_tasks.append(autotvm_task(task, args.target))
    tune_tasks(tvm_tasks, **tuning_opt)


if __name__ == '__main__':
    main()
