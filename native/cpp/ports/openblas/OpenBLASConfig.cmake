get_filename_component(_openblas_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(OpenBLAS_FOUND TRUE)
set(OpenBLAS_VERSION "0.3.33")
set(OpenBLAS_INCLUDE_DIRS "${_openblas_prefix}/include")
set(OpenBLAS_LIBRARIES "${_openblas_prefix}/lib/libopenblas.lib")
if(NOT TARGET OpenBLAS::OpenBLAS)
    add_library(OpenBLAS::OpenBLAS SHARED IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_IMPLIB "${OpenBLAS_LIBRARIES}"
        IMPORTED_LOCATION "${_openblas_prefix}/bin/libopenblas.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}")
endif()
