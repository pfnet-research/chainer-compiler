import os
import pathlib
import subprocess

import setuptools
from setuptools.command import build_ext


class CMakeExtension(setuptools.Extension):

    def __init__(self, name, build_targets, sourcedir=''):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_targets = build_targets


class CMakeBuild(build_ext.build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cuda_path = os.getenv('CUDA_PATH', '/usr/local/cuda')
        python3_executable = subprocess.check_output(['which', 'python3'])\
            .decode('utf-8').rstrip()
        cudnn_root_dir = os.getenv('CUDNN_ROOT_DIR', None)
        if cudnn_root_dir is None:
            raise RuntimeError('CUDNN_ROOT_DIR must be set.')

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCHAINER_COMPILER_ENABLE_CUDA=ON',
            '-DCHAINERX_BUILD_CUDA=ON',
            '-DCUDA_TOOLKIT_ROOT_DIR=' + cuda_path,
            '-DCHAINER_COMPILER_ENABLE_PYTHON=ON',
            '-DPYTHON_EXECUTABLE=' + python3_executable,
            '-DCHAINERX_BUILD_PYTHON=ON',
            '-DCUDNN_ROOT_DIR=' + cudnn_root_dir,
        ]
        build_args = [
            '--', '-j4',
        ]
        build_args += ext.build_targets

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', str(cwd)] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


setuptools.setup(
    name='chainer-compiler',
    version='0.0.1',
    packages=['chainer_compiler'],
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension(
        name='chainer_compiler._core',
        build_targets=['_chainer_compiler_core.so'],
        sourcedir='.')],
    cmdclass={
        'build_ext': CMakeBuild,
    }
)
