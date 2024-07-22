project(svo)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple)
# catkin_simple()

include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

list(APPEND SVO_SOURCEFILES
  src/frame_handler_mono.cpp
  src/frame_handler_base.cpp
  src/map.cpp
  src/pose_optimizer.cpp
  src/initialization.cpp
  src/reprojector.cpp
  src/io.cpp
)

list(APPEND SVO_HEADERFILES
  include/svo/frame_handler_base.h
  include/svo/frame_handler_mono.h
  include/svo/global.h
  include/svo/initialization.h
  include/svo/map.h
  include/svo/pose_optimizer.h
  include/svo/reprojector.h
  include/svo/io.h
  include/svo/svo.h
  )

################################################################################
# Create svo library
# include_directories(include ${INCLUDE_DIRS})
# cs_add_library(svo SHARED ${SVO_SOURCEFILES} ${SVO_HEADERFILES})
add_library(svo
  SHARED
  ${SVO_SOURCEFILES}
)
target_include_directories(svo 
  PUBLIC
  include
)
target_link_libraries(svo ${LINK_LIBS} svo_common vikit_common vikit_cameras rpg_common svo_img_align svo_tracker opengv)

################################################################################
# cs_install()
# cs_export()
