cmake_minimum_required(VERSION 3.10)
project(test_so)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "/usr/include/opencv4"
)

add_library(uffssd SHARED IMPORTED)
set_property(TARGET uffssd PROPERTY IMPORTED_LOCATION "/home/ballsbot/projects/test-so/libuffssd.so")

find_library(OPENCV_CORE opencv_core HINTS "/usr/lib/aarch64-linux-gnu")
find_library(OPENCV_VIDEOIO opencv_videoio HINTS "/usr/lib/aarch64-linux-gnu")

add_executable(test_so main.cpp uffssd.h)
target_link_libraries(test_so uffssd ${OPENCV_CORE} ${OPENCV_VIDEOIO}
        -Wl,--unresolved-symbols=ignore-in-shared-libs
        )
