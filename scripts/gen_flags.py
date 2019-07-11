import argparse


FLAGS = {
    'compiler_log': {
        'type': 'bool',
        'doc': 'Enables logging.'
    },
    'premissive': {
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

    'use_nvrtc': {
        'type': 'bool',
        'doc': 'Use TVM to execute operations.'
    },
    'reuse_tvm_code': {
        'type': 'bool',
        'doc': 'Reuse existing TVM code. Unsafe.'
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
        'doc': 'Dump the ONNX subgraphs'
    },

    'computation_order': {
        'type': 'std::string',
        'doc': 'The policy of computation order.'
    },
    'chen_budget': {
        'type': 'int',
    },
}


parser = argparse.ArgumentParser(description='Generate compiler option codes')
parser.add_argument('--mode')
parser.add_argument('--output')
args = parser.parse_args()

f = open(args.output)

if args.mode == 'flags.h':
    f.write("#pragma once\n")
    f.write('''
    #pragma once

    #include <string>

    namespace chainer_compiler {

    ''')
    for name, v in FLAGS:
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
    for name, v in FLAGS:
        f.write('''

        // {}
        {} g_{};
        '''.format(v['doc'], v['type'], name))
    f.write('''

    }  // namespace chainer_compiler
    ''')

elif args.mode == 'run_onnx_flags.inc':
    f.write
