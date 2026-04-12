aux_source_directory( . AUX_SRC_VIO_STEREO_VINS )

# Add all relative components of slam utility.
set( VIO_STEREO_VINS_PATH ${PROJECT_SOURCE_DIR}/../VIO_Stereo_VINS )
if( NOT TARGET lib_vio_stereo_vins_data_manager )
    add_subdirectory( ${VIO_STEREO_VINS_PATH}/src/data_manager ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_data_manager )
endif()
if( NOT TARGET lib_vio_stereo_vins_frontend )
    add_subdirectory( ${VIO_STEREO_VINS_PATH}/src/frontend ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_frontend )
endif()
if( NOT TARGET lib_vio_stereo_vins_backend )
    add_subdirectory( ${VIO_STEREO_VINS_PATH}/src/backend ${PROJECT_SOURCE_DIR}/build/lib_vio_stereo_vins_backend )
endif()

# Add general data loader.
set( SLAM_DATALOADER_PATH ${PROJECT_SOURCE_DIR}/../Slam_Dataloader )
if( NOT TARGET lib_general_data_loader )
    add_subdirectory( ${SLAM_DATALOADER_PATH}/src/data_loader ${PROJECT_SOURCE_DIR}/build/lib_general_data_loader )
endif()

add_library( lib_vio_stereo_vins ${AUX_SRC_VIO_STEREO_VINS} )
target_include_directories( lib_vio_stereo_vins PUBLIC . )
target_link_libraries( lib_vio_stereo_vins
    lib_vio_stereo_vins_data_manager
    lib_vio_stereo_vins_frontend
    lib_vio_stereo_vins_backend

    lib_general_data_loader
)
