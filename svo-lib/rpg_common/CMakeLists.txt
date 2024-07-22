project(rpg_common)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple REQUIRED)
# catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)

FILE(GLOB_RECURSE HEADERS "include/*")

list(APPEND SOURCES src/fs.cpp)

# cs_add_library(${PROJECT_NAME} ${HEADERS} ${SOURCES})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
add_library(rpg_common
  ${SOURCES}
)
target_include_directories(rpg_common 
  PUBLIC
  include
)
target_link_libraries(rpg_common ${OpenCV_LIBRARIES} glog)


# cs_install()
# cs_export()
