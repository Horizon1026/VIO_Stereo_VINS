cd build/

dataset_subdir="MH_01_easy"
# MH_01_easy
# MH_03_medium
# MH_05_difficult

# For ubuntu22.04
dataset_root_dir="/media/horizon/Documents/My_Github/Datasets/Euroc/${dataset_subdir}"
if [ -d "${dataset_root_dir}" ]; then
    ./test_vio "${dataset_root_dir}/"
else
    echo ">> Failed to find ${dataset_root_dir}/"
fi

# For windows
dataset_root_dir="/d/My_Github/Datasets/Euroc/${dataset_subdir}"
if [ -d "${dataset_root_dir}" ]; then
    ./test_vio "${dataset_root_dir}/"
else
    echo ">> Failed to find ${dataset_root_dir}/"
fi

# For windows(wsl2) ubuntu20.04
dataset_root_dir="/mnt/d/My_Github/Datasets/Euroc/${dataset_subdir}"
if [ -d "${dataset_root_dir}" ]; then
    ./test_vio "${dataset_root_dir}/"
else
    echo ">> Failed to find ${dataset_root_dir}/"
fi

cd ..