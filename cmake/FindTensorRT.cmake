include(FindPackageHandleStandardArgs)
find_package(CUDAToolkit)
find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY_INFER nvinfer
  HINTS ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TensorRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TensorRT_LIBRARY_PARSERS nvparsers
  HINTS ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TensorRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_package_handle_standard_args(
  TensorRT DEFAULT_MSG
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARY_INFER
  TensorRT_LIBRARY_INFER_PLUGIN
  TensorRT_LIBRARY_PARSERS
  TensorRT_LIBRARY_ONNXPARSER
)
mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARY_INFER
  TensorRT_LIBRARY_INFER_PLUGIN
  TensorRT_LIBRARY_PARSERS
  TensorRT_LIBRARY_ONNXPARSER
)
set(TensorRT_LIBS
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin
  TensorRT::nvparsers
  TensorRT::nvonnxparser
)
if (TensorRT_FOUND)
  add_library(TensorRT::nvinfer SHARED IMPORTED)
  set_property(TARGET TensorRT::nvinfer PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  if (WIN32)
    set_property(TARGET TensorRT::nvinfer PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY_INFER}")
  else()
    set_property(TARGET TensorRT::nvinfer PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY_INFER}")
  endif()
  add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
  if (WIN32)
    set_property(TARGET TensorRT::nvinfer_plugin PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY_INFER_PLUGIN}")
  else()
    set_property(TARGET TensorRT::nvinfer_plugin PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY_INFER_PLUGIN}")
  endif()
  add_library(TensorRT::nvparsers SHARED IMPORTED)
  if (WIN32)
    set_property(TARGET TensorRT::nvparsers PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY_PARSERS}")
  else()
    set_property(TARGET TensorRT::nvparsers PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY_PARSERS}")
  endif()
  add_library(TensorRT::nvonnxparser SHARED IMPORTED)
  if (WIN32)
    set_property(TARGET TensorRT::nvonnxparser PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY_ONNXPARSER}")
  else()
    set_property(TARGET TensorRT::nvonnxparser PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY_ONNXPARSER}")
  endif()
endif()
