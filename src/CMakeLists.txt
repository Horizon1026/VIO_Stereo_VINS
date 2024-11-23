aux_source_directory( . AUX_SRC_VIO_STEREO_VINS )

# Add all relative components of slam utility.
set( SRC_PATH ${PROJECT_SOURCE_DIR}/../VIO_Stereo_VINS/src )
if ( NOT TARGET lib_vio_stereo_vins_data_loader )
    add_subdirectory( ${SRC_PATH}/data_loader ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_data_loader )
endif()
if ( NOT TARGET lib_vio_stereo_vins_data_manager )
    add_subdirectory( ${SRC_PATH}/data_manager ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_data_manager )
endif()
if ( NOT TARGET lib_vio_stereo_vins_frontend )
    add_subdirectory( ${SRC_PATH}/frontend ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_frontend )
endif()
if ( NOT TARGET lib_vio_stereo_vins_backend )
    add_subdirectory( ${SRC_PATH}/backend ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_backend )
endif()

add_library( lib_vio_stereo_vins ${AUX_SRC_VIO_STEREO_VINS} )
target_include_directories( lib_vio_stereo_vins PUBLIC . )
target_link_libraries( lib_vio_stereo_vins
    lib_vio_stereo_vins_data_loader
    lib_vio_stereo_vins_data_manager
    lib_vio_stereo_vins_frontend
    lib_vio_stereo_vins_backend
)
