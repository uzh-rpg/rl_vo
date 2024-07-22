project(svo_common)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple REQUIRED)
# catkin_simple()
find_package(OpenCV REQUIRED)

# include(SvoSetup)
include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

set(HEADERS
  include/svo/common/camera.h
  include/svo/common/camera_fwd.h
  include/svo/common/feature_wrapper.h
  include/svo/common/frame.h
  include/svo/common/logging.h
  include/svo/common/occupancy_grid_2d.h
  include/svo/common/point.h
  include/svo/common/seed.h
  include/svo/common/transformation.h
  include/svo/common/types.h
  )

set(SOURCES
  src/frame.cpp
  src/point.cpp
  )

# cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_library(svo_common SHARED ${SOURCES})
target_include_directories(svo_common 
  PUBLIC
  include
)
target_link_libraries(svo_common ${OpenCV_LIBRARIES} vikit_common fast vikit_cameras)

################################################################################
# cs_install()
# cs_export()
