include(FindPackageHandleStandardArgs)
find_path(tensorflow_INCLUDE_DIR tensorflow/c/c_api.h PATH_SUFFIXES include)
find_library(tensorflow_LIBRARY tensorflow PATH_SUFFIXES lib lib64 lib/x64)
find_package_handle_standard_args(
  tensorflow DEFAULT_MSG
  tensorflow_INCLUDE_DIR
  tensorflow_LIBRARY
)
mark_as_advanced(
  tensorflow_INCLUDE_DIR
  tensorflow_LIBRARY
)
if (tensorflow_FOUND)
  add_library(tensorflow SHARED IMPORTED)
  set_property(TARGET tensorflow PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${tensorflow_INCLUDE_DIR}")
  if (WIN32)
    set_property(TARGET tensorflow PROPERTY IMPORTED_IMPLIB "${tensorflow_LIBRARY}")
  else()
    set_property(TARGET tensorflow PROPERTY IMPORTED_LOCATION "${tensorflow_LIBRARY}")
  endif()
endif()
