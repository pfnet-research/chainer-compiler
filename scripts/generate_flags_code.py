import argparse


FLAGS = {
    'compiler_log': {
        'type': 'bool',
        'doc': 'Enables logging.'
    },
    'permissive': {
        'type': 'bool',
        'doc': 'The compiler will accept some kinds of invalid operations to support older ONNX, etc.'
    },
    'skip_inference': {
        'type': 'bool',
        'doc': 'Skip dtype/shape inference.'
    },
    'use_cuda': {
        'type': 'bool',
        'doc': 'Use CUDA specific ops.'
    },
    'fuse_operations': {
        'type': 'bool',
        'doc': 'Fuse consecutive element-wise operations.'
    },
    'use_nvrtc': {
        'type': 'bool',
        'doc': 'Use NVRTC to execute fused operations.'
    },

    'use_tvm': {
        'type': 'bool',
        'doc': 'Use TVM to execute operations.'
    },
    'reuse_tvm_code': {
        'type': 'bool',
        'doc': 'Reuse existing TVM code.(Unsafe)'
    },
    'dump_autotvm_task_dir': {
        'type': 'std::string',
        'doc': 'Output AutoTVM tasks in this directory.'
    },
    'autotvm_log': {
        'type': 'std::string',
        'doc': 'A tuning log of AutoTVM which contains best scheduling parameters.'
    },

    'use_ngraph': {
        'type': 'bool',
        'doc': 'Use nGraph to execute operations.',
    },
    'ngraph_device': {
        'type': 'std::string',
        'doc': 'The device of nGraph (e.g., CPU and INTELGPU).'
    },

    'use_dldt': {
        'type': 'bool',
        'doc': 'Use dldt to execute operations.'
    },
    'use_dldt_fp16': {
        'type': 'bool',
        'doc': 'Use fp16 with dldt.'
    },
    'dldt_device': {
        'type': 'std::string',
        'doc': 'The device of dldt (e.g., CPU and GPU).'
    },

    'backend_name': {
        'type': 'std::string',
        'doc': 'The name of backend.'
    },

    'trace_level': {
        'type': 'int',
        'doc': 'Enables ChainerX VM trace during constant propagation.'
    },
    'reset_shape': {
        'type': 'bool',
        'doc': 'Reset all shapes.'
    },
    'reset_output_shape': {
        'type': 'bool',
        'doc': 'Reset output shapes.'
    },

    'dump_after_inference': {
        'type': 'bool',
        'doc': 'Dump the ONNX graph after inference'
    },
    'dump_after_simplification': {
        'type': 'bool',
        'doc': 'Dump the ONNX graph after simplification'
    },
    'dump_after_gradient': {
        'type': 'bool',
        'doc': 'Dump the ONNX graph after gradient'
    },
    'dump_after_fusion': {
        'type': 'bool',
        'doc': 'Dump the ONNX graph after fusion'
    },
    'dump_after_scheduling': {
        'type': 'bool',
        'doc': 'Dump the ONNX graph after scheduling'
    },
    'dump_subgraphs': {
        'type': 'bool',
        'doc': 'Dump the subgraph tree of the ONNX graph'
    },

    'computation_order': {
        'type': 'std::string',
        'doc': 'Run the specified policy of computation order (backprop only)'
    },
    'chen_budget': {
        'type': 'int',
        'doc': 'Memory budget of Chen\'s policy (in MB)'
    },
    'gt_budget': {
        'type': 'int',
        'doc': 'Memory budget of GT policy (in MB)'
    },
}


parser = argparse.ArgumentParser(description='Generate compiler option codes')
parser.add_argument('--mode')
parser.add_argument('--output')
args = parser.parse_args()

f = open(args.output, mode='w')

if args.mode == 'flags.h':
    f.write('''
#pragma once

#include <string>

namespace chainer_compiler {

''')
    for name, v in FLAGS.items():
        f.write('''
// {}
extern {} g_{};
        '''.format(v['doc'], v['type'], name))
    f.write('''

}  // namespace chainer_compiler
''')

elif args.mode == 'flags.cc':
    f.write('''
#include "compiler/flags.h"

namespace chainer_compiler {
''')
    for name, v in FLAGS.items():
        f.write('''

// {}
{} g_{};
'''.format(v['doc'], v['type'], name))

    f.write('''
struct Flags {
''')
    for name, v in FLAGS.items():
        f.write('''
{} {};
'''.format(v['type'], name))
    f.write('''
    void ApplyToGlobal() const;
};

}  // namespace chainer_compiler
''')

elif args.mode == 'compiler_flags.cc':
    f.write('''
#include "tools/compiler_flags.h"

#include <compiler/flags.h>

namespace chainer_compiler {
namespace runtime {

void AddCompilerFlags(cmdline::parser* args) {
''')
    for name, info in FLAGS.items():
        type_param = '' if info['type'] == 'bool' else '<{}>'.format(info['type'])
        def_arg = '' if info['type'] == 'bool' else ', false'
        f.write('''
    args->add{}("{}", '\\0', "{}"{});
'''.format(type_param, name, info['doc'], def_arg))

    f.write('''
}

void ApplyCompilerFlags(const cmdline::parser& args) {
''')
    for name, info in FLAGS.items():
        func = 'exist' if info['type'] == 'bool' else 'get<{}>'.format(info['type'])
        f.write('''
    g_{} = args.{}("{}");
'''.format(name, func, name))
    f.write('''
    if (args.exist("trace")) g_trace_level = 1;
    if (args.exist("verbose")) g_trace_level = 2;
}

}  // namespace runtime
}  // namespace chainer_compiler
''')
elif args.mode == 'chainer_compiler_core.cxx_args.inc':
    res = []
    for name, info in sorted(FLAGS.items()):
        res.append('{} {}'.format(info['type'], name))
    f.write(', '.join(res))
elif args.mode == 'chainer_compiler_core.apply_cxx_args.inc':
    for name, info in sorted(FLAGS.items()):
        f.write('''
        g_{} = {};
'''.format(name, name))
elif args.mode == 'chainer_compiler_core.pybind_args.inc':
    res = []
    for name, info in sorted(FLAGS.items()):
        def_val = 'false' if info['type'] == 'bool' else '""' if info['type'] == 'std::string' else '0'
        res.append('"{}"_a = {}'.format(name, def_val))
    f.write(', '.join(res))
elif args.mode == 'menoh_chainer_compiler.json_args.inc':
    res = []
    for name, info in sorted(FLAGS.items()):
        if info['type'] == 'bool':
            default = 'false'
        elif info['type'] == 'int':
            default = '0'
        elif info['type'] == 'std::string':
            default = '""'
        res.append('chainer_compiler::g_{0} = value_or<{2}>(j, "{0}", {1});'.format(name, default, info['type']))
    f.write('\n'.join(res))
elif args.mode == 'menoh_example_default_config.json':
    import json
    config = {}
    for name, info in sorted(FLAGS.items()):
        if info['type'] == 'bool':
            default = False
        elif info['type'] == 'int':
            default = 0
        elif info['type'] == 'std::string':
            default = ''
        config[name] = default
    f.write(json.dumps(config, indent=2, sort_keys=True))
else:
    raise('Invalid mode: {}'.format(args.mode))
