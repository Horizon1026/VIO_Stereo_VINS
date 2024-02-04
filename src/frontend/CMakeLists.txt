aux_source_directory( . AUX_SRC_VIO_STEREO_BASALT_FRONTEND )

# Add stereo visual frontend.
set( VISUAL_FRONTEND_PATH ${PROJECT_SOURCE_DIR}/../Visual_Frontend/src/frontend_stereo )
if ( NOT TARGET lib_visual_frontend_stereo )
    add_subdirectory( ${VISUAL_FRONTEND_PATH} ${PROJECT_SOURCE_DIR}/build/lib_visual_frontend_stereo )
endif()

# Create library.
add_library( lib_vio_stereo_vins_frontend ${AUX_SRC_VIO_STEREO_BASALT_FRONTEND} )
target_include_directories( lib_vio_stereo_vins_frontend PUBLIC . )
target_link_libraries( lib_vio_stereo_vins_frontend
    lib_visual_frontend_stereo
)
