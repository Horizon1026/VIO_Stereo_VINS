cmake_minimum_required( VERSION 3.10 )

project( test_vio_stereo_vins )

set( CMAKE_BUILD_TYPE "RelWithDegInfo" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -Wall -pthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add stack backward.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_stack_backward )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/backward ${PROJECT_SOURCE_DIR}/build/lib_stack_backward )
endif()

# Add vio stereo vins.
if ( NOT TARGET lib_vio_stereo_vins )
    add_subdirectory( src ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins )
endif()

# Add test_vio.
add_executable( test_vio
    test/test_vio.cpp
)
target_link_libraries( test_vio
    lib_vio_stereo_vins
    lib_stack_backward
)
