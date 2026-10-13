cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Target CUDA architectures")
endif()
project(GeosunPre VERSION 1.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
include(dependencies/dependencies.cmake)
find_package(CUDAToolkit REQUIRED)
add_executable(geosunpre
  src/app/main.cpp src/app/pipeline.cpp src/app/runtime.cpp
  src/io/scene.cpp src/io/cache.cpp src/io/las.cpp src/io/colmap.cpp
  src/geometry/camera.cpp src/geometry/model.cpp
  src/frontend/inference.cpp src/frontend/kernels.cu src/frontend/preprocessing.cu src/frontend/frontend.cpp
  src/sfm/reconstruction.cpp src/sfm/optimization.cpp src/sfm/photometric.cpp
  src/alignment/alignment.cpp)
execute_process(COMMAND git describe --always --dirty WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GEOSUN_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GEOSUN_REVISION)
  set(GEOSUN_REVISION "source-archive")
endif()
string(TIMESTAMP GEOSUN_BUILD_TIME "%Y-%m-%dT%H:%M:%SZ" UTC)
target_compile_definitions(geosunpre PRIVATE GEOSUN_VERSION="${PROJECT_VERSION}"
  GEOSUN_REVISION="${GEOSUN_REVISION}" GEOSUN_BUILD_TIME="${GEOSUN_BUILD_TIME}")
target_include_directories(geosunpre PRIVATE src ${tensorrt_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_include_directories(geosunpre SYSTEM PRIVATE ${opencv_BINARY_DIR} ${CMAKE_BINARY_DIR}
  ${opencv_SOURCE_DIR}/modules/core/include ${opencv_SOURCE_DIR}/modules/imgproc/include
  ${opencv_SOURCE_DIR}/modules/imgcodecs/include ${opencv_SOURCE_DIR}/modules/calib3d/include
  ${opencv_SOURCE_DIR}/modules/features2d/include ${opencv_SOURCE_DIR}/modules/flann/include)
target_link_libraries(geosunpre PRIVATE Ceres::ceres Eigen3::Eigen nlohmann_json::nlohmann_json
  nanoflann::nanoflann opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d CUDA::cudart)
if(WIN32)
  target_compile_definitions(geosunpre PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(geosunpre PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
  target_link_libraries(geosunpre PRIVATE psapi)
else()
  target_link_libraries(geosunpre PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(geosunpre PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS geosunpre RUNTIME DESTINATION .)
install(DIRECTORY docs DESTINATION .)
install(FILES README.md DESTINATION .)
