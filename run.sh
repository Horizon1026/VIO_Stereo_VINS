cd build/

dataset_subdir="MH_01_easy"
# MH_01_easy
# MH_03_medium
# MH_05_difficult

dataset_root_dir="/media/horizon/Documents/My_Github/Datasets/${dataset_subdir}"
if [ -d "${dataset_root_dir}" ]; then
    ./test_vio "${dataset_root_dir}/"
else
    echo ">> Failed to find ${dataset_root_dir}/"
fi

dataset_root_dir="/d/My_Github/Datasets/${dataset_subdir}"
if [ -d "${dataset_root_dir}" ]; then
    ./test_vio "${dataset_root_dir}/"
else
    echo ">> Failed to find ${dataset_root_dir}/"
fi

cd ..