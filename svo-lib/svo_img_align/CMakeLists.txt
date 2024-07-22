project(svo_img_align)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple)
# catkin_simple()

include(../svo_cmake/cmake/Modules/SvoSetup.cmake)

set(HEADERS_CPU
  include/svo/img_align/sparse_img_align_base.h
  include/svo/img_align/sparse_img_align.h
  )

set(SOURCES_CPU
  src/sparse_img_align_base.cpp
  src/sparse_img_align.cpp
  )

# cs_add_library(${PROJECT_NAME} ${SOURCES_CPU} ${HEADERS_CPU})
# target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})
add_library(svo_img_align
  SHARED
  ${SOURCES_CPU}
)
target_include_directories(svo_img_align 
  PUBLIC
  include
)
target_link_libraries(svo_img_align ${LINK_LIBS} svo_common svo_direct vikit_common vikit_solver)

# cs_install()
# cs_export()
