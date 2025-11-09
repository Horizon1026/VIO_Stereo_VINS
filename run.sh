cd ./build

# Define the datasets to be processed using space-separated strings.
dataset_root_dirs=\
" /mnt/d/My_Github/Datasets/Euroc"\
" /home/jiapengli/Desktop/dataset/Euroc"\
" /mnt/d/robotic_datasets/slam/Euroc"\
" /media/horizon/Database/robotic_datasets/slam/Euroc"

dataset_subdirs=\
" MH_01_easy"\
" MH_02_easy"\
" MH_03_medium"\
" MH_04_difficult"\
" MH_05_difficult"\
" V1_01_easy"\
" V1_02_medium"\
" V1_03_difficult"\
" V2_01_easy"\
" V2_02_medium"\
" V2_03_difficult"

# Define the log root path.
log_output_root_dir="../../Workspace/output/"

# Iterate over each dataset using set and shift.
set -- $dataset_subdirs
for dataset_subdir do
    echo ">> Processing dataset: ${dataset_subdir}"

    # Try to find the dataset in the dataset_root_dirs.
    dataset_path=""
    for dataset_root_dir in $dataset_root_dirs; do
        if [ -d "${dataset_root_dir}/${dataset_subdir}" ]; then
            dataset_path="${dataset_root_dir}/${dataset_subdir}"
            break
        fi
    done

    # If the dataset is found, run the VIO test.
    if [ -d "${dataset_path}" ]; then
        echo ">> Found dataset at: ${dataset_path}"
        echo ">> Running VIO test..."

        # Try to create the log output directory.
        log_output_dir="${log_output_root_dir}/${dataset_subdir}"
        if [ ! -d "${log_output_dir}" ]; then
            mkdir -p "${log_output_dir}"
        fi

        # Run VIO test.
        if ./test_vio "${dataset_path}/" "${log_output_dir}/"; then
            echo ">> VIO test completed successfully for ${dataset_subdir}"
        else
            echo ">> VIO test failed for ${dataset_subdir}"
        fi
    else
        echo ">> Dataset not found: ${dataset_path}, skip it."
    fi
done

cd ..
