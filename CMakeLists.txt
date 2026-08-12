cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(awakening LANGUAGES CXX C)
include(CheckCXXSourceCompiles)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

#修改以下option需要rebuild清除缓存才能生效
option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_SENTRY "Enable Sentry"   ON)
option(BUILD_RADAR "Enable Radar"   ON)
option(BUILD_RUNTIME_TESTS "Build runtime test executables by default" ON)
option(BUILD_WITH_PCH "Enable precompiled headers for C++ targets" ON)
option(BUILD_WITH_Daedalus "Enable Daedalus interface" ON)
option(BUILD_WITH_RERUN "Enable Rerun visualization when the SDK is available" ON)
option(BUILD_WITH_ROS2 "Enable ROS2 targets when the ROS2 environment is available" ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(AWAKENING_PCH_HEADER ${PROJECT_SOURCE_DIR}/src/pch.hpp)
include(AwakeningTargets)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(HikSDK QUIET)
find_package(MvSDK QUIET)
set(DAHENG_SDK_ROOT "" CACHE PATH "Daheng Galaxy SDK installation prefix")
if(DAHENG_SDK_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${DAHENG_SDK_ROOT}")
endif()
find_package(DahengSDK QUIET)
set(BUILD_DEFINITIONS "")
set(BUILD_LIBS "")
if(HikSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_HikSDK)
    list(APPEND BUILD_LIBS HikSDK::HikSDK)
endif()

if(MvSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_MvSDK)
    list(APPEND BUILD_LIBS MvSDK::MvSDK)
endif()

if(DahengSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_DahengSDK)
    list(APPEND BUILD_LIBS DahengSDK::DahengSDK)
endif()

set(RERUN_FOUND FALSE)
if(BUILD_WITH_RERUN)
    find_package(rerun_sdk CONFIG QUIET)
    if(TARGET rerun_sdk)
        set(RERUN_FOUND TRUE)
        message(STATUS "Rerun SDK found, enabling the Rerun visualization backend")
    else()
        message(WARNING "Rerun SDK not found; Web/ROS2 visualization remains available")
    endif()
endif()



if(RERUN_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_RERUN)
endif()
if(BUILD_WITH_Daedalus)
    check_cxx_source_compiles(
        "#include <expected>
         int main() { std::expected<int, int> value = 1; return *value; }"
        AWAKENING_HAS_STD_EXPECTED
    )
    if(AWAKENING_HAS_STD_EXPECTED)
        list(APPEND BUILD_DEFINITIONS USE_Daedalus)
    else()
        message(WARNING "Daedalus disabled: compiler does not provide <expected>")
    endif()
endif()

if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
    list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()

if(BUILD_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
    if(TENSORRT_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${TENSORRT_ROOT}")
    endif()
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIR})
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    list(APPEND BUILD_DEFINITIONS USE_TRT)
    list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart)
    if(TARGET CUDA::nppicc)
        list(APPEND BUILD_LIBS CUDA::nppicc)
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/param_deliver.h.in
    ${CMAKE_BINARY_DIR}/src/param_deliver.h
    @ONLY
)
set(AWAKENING_COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${CMAKE_BINARY_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)
awakening_add_library(relink SOURCES src/relink.cpp)
set(UTILS_SOURCE_GLOBS src/utils/*.cpp 3rdparty/*.cpp 3rdparty/*.c)
if(BUILD_WITH_TRT)
    list(APPEND UTILS_SOURCE_GLOBS src/utils/*.cu)
endif()
awakening_add_library(${PROJECT_NAME}_utils GLOB ${UTILS_SOURCE_GLOBS} INCLUDES ${AWAKENING_COMMON_INCLUDES} DEFINITIONS ${BUILD_DEFINITIONS} LINKS
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    spdlog::spdlog
    TBB::tbb
    Boost::boost
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
)
file(GLOB_RECURSE AWAKENING_UTILS_SOURCES CONFIGURE_DEPENDS ${UTILS_SOURCE_GLOBS})
list(FILTER AWAKENING_UTILS_SOURCES EXCLUDE REGEX "/backward-cpp/(test|test_package)/")
set_property(TARGET ${PROJECT_NAME}_utils PROPERTY SOURCES ${AWAKENING_UTILS_SOURCES})
awakening_add_library(${PROJECT_NAME}_auto_aim GLOB src/tasks/auto_aim/*.cpp LINKS ${PROJECT_NAME}_utils)
awakening_add_library(${PROJECT_NAME}_auto_buff GLOB src/tasks/auto_buff/*.cpp LINKS ${PROJECT_NAME}_utils)
awakening_add_library(${PROJECT_NAME}_web SOURCES src/tasks/base/web.cpp LINKS
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
)
awakening_add_library(${PROJECT_NAME}_vslam GLOB src/tasks/vslam/*.cpp)

if(RERUN_FOUND)
    # The distributed C++ SDK is static/non-PIC, so keep it at executable link scope.
    awakening_add_library(${PROJECT_NAME}_rerun TYPE STATIC SOURCES src/_rerun/recorder.cpp DEFINITIONS USE_RERUN
        LINKS rerun_sdk
        PRIVATE_LINKS ${PROJECT_NAME}_auto_aim ${PROJECT_NAME}_auto_buff
    )
endif()

set(ROS2_FULL_FOUND FALSE)
if(BUILD_WITH_ROS2)
    set(ROS2_PACKAGES
        ament_cmake
        rmw
        rosidl_typesupport_cpp
        rosidl_typesupport_interface
        rclcpp
        geometry_msgs
        tf2_ros
        tf2_geometry_msgs
        sensor_msgs
        visualization_msgs
        nav_msgs
    )
    foreach(pkg IN LISTS ROS2_PACKAGES)
        find_package(${pkg} QUIET)
    endforeach()

    set(ROS2_FULL_FOUND TRUE)
    foreach(pkg IN LISTS ROS2_PACKAGES)
        if(NOT ${pkg}_FOUND)
            set(ROS2_FULL_FOUND FALSE)
            message(WARNING "ROS2 package ${pkg} not found")
        endif()
    endforeach()
else()
    message(STATUS "ROS2 targets disabled by BUILD_WITH_ROS2=OFF")
endif()

if(ROS2_FULL_FOUND)
    message(STATUS "ROS2 full environment found, compiling ROS2 targets ...")
    list(APPEND BUILD_DEFINITIONS USE_ROS2)

    set(ROS2_INCLUDES)
    foreach(pkg IN LISTS ROS2_PACKAGES)
        if(TARGET ${pkg}::${pkg})
            list(APPEND ROS2_INCLUDES ${${pkg}_INCLUDE_DIRS})
        endif()
    endforeach()
    include_directories(${ROS2_INCLUDES})
    awakening_add_library(${PROJECT_NAME}_rcl EXCLUDE_FROM_ALL GLOB src/_rcl/*.cpp LINKS ${PROJECT_NAME}_utils)
    ament_target_dependencies(${PROJECT_NAME}_rcl  ${ROS2_PACKAGES})
    target_compile_definitions(${PROJECT_NAME}_rcl  PUBLIC ${BUILD_DEFINITIONS})
    if(BUILD_SENTRY)
    awakening_add_library(${PROJECT_NAME}_sentry_brain EXCLUDE_FROM_ALL GLOB src/tasks/sentry_brain/*.cpp LINKS
        ${PROJECT_NAME}_rcl
        ${PROJECT_NAME}_auto_aim
    )
    endif()
    if(BUILD_RADAR)
    # find_package(small_gicp REQUIRED)
    find_package(Open3D REQUIRED)
    awakening_add_library(${PROJECT_NAME}_radar_detect EXCLUDE_FROM_ALL GLOB src/tasks/radar_detect/*.cpp LINKS
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
        Open3D::Open3D
    )
    awakening_add_library(${PROJECT_NAME}_radar_io EXCLUDE_FROM_ALL GLOB src/tasks/radar_io/*.cpp LINKS
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
    )
    endif()
else()
    message(WARNING "ROS2 dependencies incomplete, skipping ROS2 targets ...")
endif()


set(STANDARD_LINKS ${PROJECT_NAME}_auto_aim ${PROJECT_NAME}_auto_buff ${PROJECT_NAME}_web)
set(VO_LINKS ${PROJECT_NAME}_vslam)
if(ROS2_FULL_FOUND)
    list(APPEND STANDARD_LINKS ${PROJECT_NAME}_rcl)
    list(APPEND VO_LINKS ${PROJECT_NAME}_rcl)
endif()

awakening_add_executable(standard SOURCES src/runtime/standard.cpp LINKS ${STANDARD_LINKS})
awakening_add_executable(control_test TOOL SOURCES src/runtime/test/control_test.cpp LINKS ${PROJECT_NAME}_web)
awakening_add_executable(delay_test TOOL SOURCES src/runtime/test/delay_test.cpp)
awakening_add_executable(vo SOURCES src/runtime/vo.cpp LINKS ${VO_LINKS})

if(BUILD_RADAR)
    awakening_add_executable(radar_detect TOOL SOURCES src/runtime/radar_detect.cpp LINKS ${PROJECT_NAME}_radar_detect ${PROJECT_NAME}_radar_io)
    awakening_add_executable(livox_v1_driver TOOL SOURCES src/runtime/livox_v1_driver.cpp LINKS ${PROJECT_NAME}_radar_detect)
    awakening_add_executable(radar_cal TOOL SOURCES src/runtime/radar_cal.cpp LINKS ${PROJECT_NAME}_radar_detect)
endif()

if(ROS2_FULL_FOUND AND BUILD_SENTRY)
    awakening_add_executable(nav TOOL SOURCES src/runtime/nav.cpp LINKS ${PROJECT_NAME}_rcl ${PROJECT_NAME}_utils)
    awakening_add_executable(sentry SOURCES src/runtime/sentry.cpp LINKS
        ${PROJECT_NAME}_rcl
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_auto_aim
        ${PROJECT_NAME}_auto_buff
        ${PROJECT_NAME}_web
        ${PROJECT_NAME}_sentry_brain
    )
    awakening_add_executable(map_edit TOOL SOURCES src/runtime/map_edit.cpp)
endif()
