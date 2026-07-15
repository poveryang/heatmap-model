cmake_minimum_required(VERSION 3.10)
project(YoloBarcodeDetector)

set(TENGINE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/tengine)

if (TARGET_PLATFORM MATCHES "imx8plus")
    message(STATUS "Target architecture: imx8plus")
    set(TENGINE_LIB_DIR ${TENGINE_DIR}/lib/aarch64)
    set(OpenCV_DIR /opt/libs_aarch64/opencv_aarch64/lib/cmake/opencv4)
elseif (TARGET_PLATFORM STREQUAL "x86")
    message(STATUS "Target architecture: x86")
    set(TENGINE_LIB_DIR ${TENGINE_DIR}/lib/x86)
    set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
endif ()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs PATHS ${OpenCV_DIR})
file(GLOB TENGINE_LIBS ${TENGINE_LIB_DIR}/*.so)

add_library(YOLO_DETECTOR STATIC src/yolo_detector.cpp)
target_include_directories(YOLO_DETECTOR PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TENGINE_DIR}/include
)
target_link_libraries(YOLO_DETECTOR PUBLIC
    ${TENGINE_LIBS}
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

add_executable(YOLO-DETECTOR-TEST test/test_yolo.cpp)
target_link_libraries(YOLO-DETECTOR-TEST PRIVATE YOLO_DETECTOR)

install(TARGETS YOLO_DETECTOR ARCHIVE DESTINATION lib)
