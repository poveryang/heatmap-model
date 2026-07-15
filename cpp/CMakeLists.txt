CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(HMAP)


SET(TENGINE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/tengine)

# Configure tengine library and opencv library
IF (TARGET_PLATFORM MATCHES "imx8plus")
    MESSAGE(STATUS "Target architecture: imx8plus")
    SET(TENGINE_LIB_DIR ${TENGINE_DIR}/lib/aarch64)
    SET(OpenCV_DIR /opt/libs_aarch64/opencv_aarch64/lib/cmake/opencv4)
ELSEIF (TARGET_PLATFORM STREQUAL "x86")
    MESSAGE(STATUS "Target architecture: x86")
    SET(TENGINE_LIB_DIR ${TENGINE_DIR}/lib/x86)
    SET(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
ENDIF ()

IF (NOT FOUND_OPENCV)
    MESSAGE(STATUS "OpenCV not found in libHMAP, use OpenCV_DIR: ${OpenCV_DIR}")
    FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs PATHS ${OpenCV_DIR})
ENDIF ()
SET(HMAP_OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs)

# Build hmap_generator library
ADD_COMPILE_OPTIONS(-fPIC)
ADD_LIBRARY(${PROJECT_NAME} STATIC src/hmap_generator.cpp)
ADD_LIBRARY(YOLO_DETECTOR STATIC src/yolo_detector.cpp)

# Include directories
SET(TENGINE_INC_DIR ${TENGINE_DIR}/include)
SET(HMAP_INC_DIR ${PROJECT_SOURCE_DIR}/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
        ${HMAP_INC_DIR}
        ${TENGINE_INC_DIR}
        )
TARGET_INCLUDE_DIRECTORIES(YOLO_DETECTOR PUBLIC
        ${HMAP_INC_DIR}
        ${TENGINE_INC_DIR}
        )

# Link libraries
FILE(GLOB TENGINE_LIBS ${TENGINE_LIB_DIR}/*.so)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC
        ${TENGINE_LIBS}
        ${HMAP_OPENCV_LIBS}
        )
TARGET_LINK_LIBRARIES(YOLO_DETECTOR PUBLIC
        ${TENGINE_LIBS}
        ${HMAP_OPENCV_LIBS}
        )


# Install library
INSTALL(TARGETS ${PROJECT_NAME} YOLO_DETECTOR
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# Build test
ADD_EXECUTABLE(${PROJECT_NAME}-TEST "./test/test_hmap.cpp")
TARGET_LINK_LIBRARIES(${PROJECT_NAME}-TEST PUBLIC
        ${PROJECT_NAME}
        ${HMAP_OPENCV_LIBS}
        )

ADD_EXECUTABLE(YOLO-DETECTOR-TEST "./test/test_yolo.cpp")
TARGET_LINK_LIBRARIES(YOLO-DETECTOR-TEST PUBLIC
        YOLO_DETECTOR
        ${HMAP_OPENCV_LIBS}
        )
