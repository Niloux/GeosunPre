cmake_minimum_required(VERSION 3.24)
# Match ampere_plus engines: SASS for Ampere/Ada/Blackwell, compute_80 PTX for other GPUs >= 8.0.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86-real;89-real;120-real" CACHE STRING "Target CUDA architectures")
endif()
project(GeosunPre VERSION 1.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
include(dependencies/dependencies.cmake)
find_package(CUDAToolkit REQUIRED)
set(GEOSUN_SOURCES
  src/app/main.cpp src/app/pipeline.cpp src/app/runtime.cpp
  src/io/scene.cpp src/io/cache.cpp src/io/las.cpp src/io/colmap.cpp
  src/geometry/camera.cpp src/geometry/model.cpp src/geometry/supervision.cu
  src/frontend/inference.cpp src/frontend/kernels.cu src/frontend/preprocessing.cu src/frontend/frontend.cpp
  src/sfm/reconstruction.cpp src/sfm/optimization.cpp src/sfm/photometric.cpp src/sfm/structure.cpp
  src/alignment/alignment.cpp)
# Keep changing build metadata out of the application's compile definitions.
file(GLOB_RECURSE GEOSUN_BUILD_INPUTS CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*" "${CMAKE_SOURCE_DIR}/cmake/*"
  "${CMAKE_SOURCE_DIR}/dependencies/*.cmake")
list(APPEND GEOSUN_BUILD_INPUTS "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_SOURCE_DIR}/CMakePresets.json" "${CMAKE_BINARY_DIR}/CMakeCache.txt")
execute_process(COMMAND git symbolic-ref -q HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GEOSUN_GIT_REF OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
foreach(git_path HEAD index packed-refs ${GEOSUN_GIT_REF})
  execute_process(COMMAND git rev-parse --git-path "${git_path}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE git_file
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE git_result)
  if(git_result EQUAL 0)
    get_filename_component(git_file "${git_file}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(EXISTS "${git_file}")
      list(APPEND GEOSUN_BUILD_INPUTS "${git_file}")
    endif()
  endif()
endforeach()
set(GEOSUN_BUILD_INFO "${CMAKE_BINARY_DIR}/generated/build_info.cpp")
add_custom_command(OUTPUT "${GEOSUN_BUILD_INFO}"
  COMMAND "${CMAKE_COMMAND}" "-DSOURCE_ROOT=${CMAKE_SOURCE_DIR}"
    "-DOUTPUT=${GEOSUN_BUILD_INFO}" "-DVERSION=${PROJECT_VERSION}"
    -P "${CMAKE_SOURCE_DIR}/cmake/build-info.cmake"
  DEPENDS ${GEOSUN_BUILD_INPUTS} VERBATIM)
# configs/default.json is the only copy of the default settings; editing it reconfigures.
set(GEOSUN_DEFAULTS "${CMAKE_SOURCE_DIR}/configs/default.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GEOSUN_DEFAULTS}")
file(READ "${GEOSUN_DEFAULTS}" GEOSUN_DEFAULT_SETTINGS)
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/default_settings.cpp" CONTENT [=[
namespace geosun {
extern const char DefaultSettings[] = R"json(@GEOSUN_DEFAULT_SETTINGS@)json";
}  // namespace geosun
]=] @ONLY)
add_executable(geosunpre ${GEOSUN_SOURCES} "${GEOSUN_BUILD_INFO}"
  "${CMAKE_BINARY_DIR}/generated/default_settings.cpp")
target_include_directories(geosunpre PRIVATE src ${tensorrt_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_include_directories(geosunpre SYSTEM PRIVATE ${opencv_BINARY_DIR} ${CMAKE_BINARY_DIR}
  ${opencv_SOURCE_DIR}/modules/core/include ${opencv_SOURCE_DIR}/modules/imgproc/include
  ${opencv_SOURCE_DIR}/modules/imgcodecs/include ${opencv_SOURCE_DIR}/modules/calib3d/include
  ${opencv_SOURCE_DIR}/modules/features2d/include ${opencv_SOURCE_DIR}/modules/flann/include)
target_link_libraries(geosunpre PRIVATE Ceres::ceres Eigen3::Eigen nlohmann_json::nlohmann_json
  nanoflann::nanoflann opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d CUDA::cudart)
if(WIN32)
  target_compile_definitions(geosunpre PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(geosunpre PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  target_link_libraries(geosunpre PRIVATE psapi)
else()
  target_link_libraries(geosunpre PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(geosunpre PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS geosunpre RUNTIME DESTINATION .)
install(DIRECTORY docs DESTINATION .)
install(FILES README.md DESTINATION .)
