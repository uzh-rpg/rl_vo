project(svo_tracker)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple REQUIRED)
# catkin_simple()

include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

set(HEADERS
  include/svo/tracker/feature_tracking_types.h
  include/svo/tracker/feature_tracking_utils.h
  include/svo/tracker/feature_tracking_viz.h
  include/svo/tracker/feature_tracker.h
  )

set(SOURCES
  src/feature_tracking_types.cpp
  src/feature_tracking_utils.cpp
  src/feature_tracking_viz.cpp
  src/feature_tracker.cpp
  )

# cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_library(svo_tracker
  SHARED
  ${SOURCES}
)
target_include_directories(svo_tracker 
  PUBLIC
  include
)
target_link_libraries(svo_tracker svo_common svo_direct)

# ################################################################################
# cs_install()
# cs_export()

