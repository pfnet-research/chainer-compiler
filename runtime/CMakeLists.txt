include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_xchainer_runtime
  xchainer.cc
  )

add_library(oniku_xchainer_runtime_main
  xchainer_main.cc
  )
