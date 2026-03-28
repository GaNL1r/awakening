cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(awakening LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")


option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       OFF)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(HikSDK REQUIRED)
find_package(TBB REQUIRED)
find_package(wust_vl REQUIRED)

set(BUILD_DEFINITIONS "")
set(BUILD_LIBS "")


if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
    list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()

if(BUILD_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-10.6.0.26")
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIR})
    include_directories(/usr/local/cuda/include)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda_infer ${CMAKE_BINARY_DIR}/cuda_infer_build)
    list(APPEND BUILD_DEFINITIONS USE_TRT)
    list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart cuda_infer)
endif()

if(BUILD_WITH_NCNN)
    find_package(ncnn REQUIRED)
    list(APPEND BUILD_DEFINITIONS USE_NCNN)
    list(APPEND BUILD_LIBS ncnn)
endif()

if(BUILD_WITH_ORT)
    set(ort_root_path "/home/hy/onnxruntime-linux-x64-gpu-1.22.0")
    find_package(Ort REQUIRED)
    include_directories(${Ort_INCLUDE_DIR})
    list(APPEND BUILD_DEFINITIONS USE_ORT)
    list(APPEND BUILD_LIBS ${Ort_LIB})
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${BUILD_DEFINITIONS})
target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    wust_vl::wust_vl
    HikSDK::HikSDK
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    TBB::tbb
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
)
function(add_common_executable exe_name src_file)
    add_executable(${exe_name} ${src_file})
    target_link_libraries(${exe_name}  ${PROJECT_NAME})
endfunction()
add_common_executable(a test/a.cpp)