cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(awakening LANGUAGES CXX C)


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
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(AWAKENING_PCH_HEADER ${PROJECT_SOURCE_DIR}/src/pch.hpp)

function(awakening_enable_pch target_name)
    if(BUILD_WITH_PCH AND COMMAND target_precompile_headers)
        target_precompile_headers(
            ${target_name}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${AWAKENING_PCH_HEADER}>"
        )
    endif()
endfunction()


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(HikSDK REQUIRED)
find_package(MvSDK REQUIRED)
find_package(spdlog REQUIRED)


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
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    list(APPEND BUILD_DEFINITIONS USE_TRT)
    list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart  /usr/local/cuda/lib64/libnppicc.so)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/param_deliver.h.in  
    ${CMAKE_BINARY_DIR}/src/param_deliver.h            
    @ONLY
)
add_library(relink SHARED src/relink.cpp)
awakening_enable_pch(relink)
include_directories(${PROJECT_NAME}_utils PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${CMAKE_BINARY_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
file(GLOB_RECURSE UTILS_SOURCES  src/utils/*.cpp 3rdparty/*.cpp 3rdparty/*.c)
if(BUILD_WITH_TRT)
    file(GLOB_RECURSE UTILS_CUDA_SOURCES  src/utils/*.cu)
    list(APPEND UTILS_SOURCES ${UTILS_CUDA_SOURCES})
endif()
add_library(${PROJECT_NAME}_utils SHARED ${UTILS_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_utils)
target_compile_definitions(${PROJECT_NAME}_utils PUBLIC ${BUILD_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_utils
    HikSDK::HikSDK
    MvSDK::MvSDK
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
    
)
file(GLOB_RECURSE AUTO_AIM_SOURCES  src/tasks/auto_aim/*.cpp)
add_library(${PROJECT_NAME}_auto_aim SHARED ${AUTO_AIM_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_auto_aim)
target_link_libraries(${PROJECT_NAME}_auto_aim
     ${PROJECT_NAME}_utils
)
file(GLOB_RECURSE AUTO_BUFF_SOURCES  src/tasks/auto_buff/*.cpp)
add_library(${PROJECT_NAME}_auto_buff SHARED ${AUTO_BUFF_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_auto_buff)
target_link_libraries(${PROJECT_NAME}_auto_buff
     ${PROJECT_NAME}_utils
)
add_library(${PROJECT_NAME}_web SHARED src/tasks/base/web.cpp)
awakening_enable_pch(${PROJECT_NAME}_web)
target_link_libraries(${PROJECT_NAME}_web
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
)

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
    file(GLOB_RECURSE RCL_SOURCES  src/_rcl/*.cpp)
    add_library(${PROJECT_NAME}_rcl SHARED EXCLUDE_FROM_ALL ${RCL_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_rcl)
    target_link_libraries(${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
    )
    ament_target_dependencies(${PROJECT_NAME}_rcl  ${ROS2_PACKAGES})
    target_compile_definitions(${PROJECT_NAME}_rcl  PUBLIC ${BUILD_DEFINITIONS})
    if(BUILD_SENTRY)
    file(GLOB_RECURSE SENTRY_BRAIN_SOURCES  src/tasks/sentry_brain/*.cpp)
    add_library(${PROJECT_NAME}_sentry_brain SHARED EXCLUDE_FROM_ALL ${SENTRY_BRAIN_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_sentry_brain)
    target_link_libraries(${PROJECT_NAME}_sentry_brain
        ${PROJECT_NAME}_rcl
        ${PROJECT_NAME}_auto_aim 
    )
    endif()
    if(BUILD_RADAR)
    # find_package(small_gicp REQUIRED)
    find_package(Open3D REQUIRED)
    file(GLOB_RECURSE RADAR_DETECT_SOURCES  src/tasks/radar_detect/*.cpp)
    add_library(${PROJECT_NAME}_radar_detect SHARED EXCLUDE_FROM_ALL ${RADAR_DETECT_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_radar_detect)
    target_link_libraries(${PROJECT_NAME}_radar_detect
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
        Open3D::Open3D
    )
    file(GLOB_RECURSE RADAR_IO_SOURCES  src/tasks/radar_io/*.cpp)
    add_library(${PROJECT_NAME}_radar_io SHARED EXCLUDE_FROM_ALL ${RADAR_IO_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_radar_io)
    target_link_libraries(${PROJECT_NAME}_radar_io
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
    )
    endif()
else()
    message(WARNING "ROS2 dependencies incomplete, skipping ROS2 targets ...")
endif()


function(add_common_executable exe_name src_file)
    add_executable(${exe_name} ${src_file})
    awakening_enable_pch(${exe_name})
    target_link_libraries(${exe_name}  ${PROJECT_NAME}_utils relink)
endfunction()

function(add_tool_executable exe_name src_file)
    if(BUILD_RUNTIME_TESTS)
        add_executable(${exe_name} ${src_file})
    else()
        add_executable(${exe_name} EXCLUDE_FROM_ALL ${src_file})
    endif()
    awakening_enable_pch(${exe_name})
    target_link_libraries(${exe_name} ${PROJECT_NAME}_utils relink)
endfunction()

add_common_executable(standard src/runtime/standard.cpp)
add_tool_executable(control_test src/runtime/test/control_test.cpp)
add_tool_executable(delay_test src/runtime/test/delay_test.cpp)


target_link_libraries(standard
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
    ${PROJECT_NAME}_web
)
target_link_libraries(control_test
    ${PROJECT_NAME}_web
)


if(BUILD_RADAR)
add_tool_executable(radar_detect src/runtime/radar_detect.cpp)
target_link_libraries(radar_detect
    ${PROJECT_NAME}_radar_detect
    ${PROJECT_NAME}_radar_io
)
add_tool_executable(livox_v1_driver src/runtime/livox_v1_driver.cpp)
target_link_libraries(livox_v1_driver
    ${PROJECT_NAME}_radar_detect
)
add_tool_executable(radar_cal src/runtime/radar_cal.cpp)
target_link_libraries(radar_cal
    ${PROJECT_NAME}_radar_detect
)
endif()
if(ROS2_FULL_FOUND)
add_tool_executable(camera_to_ros2 src/runtime/test/camera_to_ros2.cpp)
target_link_libraries(camera_to_ros2
    ${PROJECT_NAME}_rcl
)
target_link_libraries(standard
    ${PROJECT_NAME}_rcl
)

if(BUILD_SENTRY)
add_tool_executable(nav src/runtime/nav.cpp)
add_common_executable(sentry src/runtime/sentry.cpp)
add_tool_executable(map_edit src/runtime/map_edit.cpp)

target_link_libraries(nav
    ${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
)
target_link_libraries(sentry
    ${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
    ${PROJECT_NAME}_web
    ${PROJECT_NAME}_sentry_brain
)

endif()
endif()
