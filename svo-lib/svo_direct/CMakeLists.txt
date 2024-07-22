project(svo_direct)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple REQUIRED)
# catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV REQUIRED)

include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

set(HEADERS
  include/svo/direct/patch_warp.h
  include/svo/direct/patch_score.h
  include/svo/direct/patch_utils.h
  include/svo/direct/feature_alignment.h
  include/svo/direct/feature_detection_types.h
  include/svo/direct/feature_detection.h
  include/svo/direct/feature_detection_utils.h
  include/svo/direct/matcher.h
  include/svo/direct/depth_filter.h
  include/svo/direct/depth_estimation.h
  include/svo/direct/elder_zucker.h
)

set(SOURCES
  src/patch_warp.cpp
  src/feature_alignment.cpp
  src/feature_detection.cpp
  src/feature_detection_utils.cpp
  src/matcher.cpp
  src/depth_filter.cpp
  src/depth_estimation.cpp
  src/elder_zucker.cpp
  )

# cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
add_library(svo_direct
  SHARED
  ${SOURCES}
)
target_include_directories(svo_direct 
  PUBLIC
  include
)
target_link_libraries(svo_direct svo_common vikit_solver)


################################################################################
# cs_install()
# cs_export()

