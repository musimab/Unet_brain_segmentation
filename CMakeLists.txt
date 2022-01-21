cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(U_NET)
SET(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH /home/mustafa/Downloads/libtorch)
find_package(Torch REQUIRED)
add_executable(unet main_.cpp brainUnet.cpp)

include_directories(
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(unet ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET unet PROPERTY CXX_STANDARD 17)
