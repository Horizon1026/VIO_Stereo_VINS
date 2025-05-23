# VIO_Stereo_VINS
A simple visual-inertial odometry reconstructed from vins-fusion.

# Components
- [x] Data manager.
    - [x] Local map.
    - [x] Visualizor.
- [x] Data loader.
- [x] Frontend.
    - [x] Stereo visual frontend (Base on repo: Visual_Frontend).
- [x] Backend.
    - [x] Backend initialization.
    - [x] Backend optimization.
    - [x] Backend marginalization.
- [x] Log record.

# Dependence
- Slam_Utility
- Feature_Detector
- Feature_Tracker
- Sensor_Model
- Vision_Geometry
- Image_Processor
- Slam_Solver
- Visual_Frontend
- Binary_Data_Log
- Visualizor2D
- Visualizor3D

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 这是为了学习开源的 VINS-FUSION 而创建的用于复现/魔改 paper 的代码仓库，欢迎一起交流学习，不同意商用；
- 依赖仓库为各个“积木”仓库，陈列在本文 Dependence 中。如需编译（标准 cmake 编译流程），需要拉下所有依赖仓库的源码。如需运行，需要更改 run.sh 中的数据集路径，在 test_vio.cpp 中配置参数，并在各个“积木”仓库的同一路径下创建 ./Workspace/output/ 文件夹，用于保存输出文件；
- 不依赖 opencv 和开源求解器（如 ceres，g2o，gtsam之类），是自己实现的 LM 求解器，详见 Slam_Solver 仓库；
- 这不是为了实机部署，而是为了算法学习。为了尽可能仅保留算法部分，牺牲了一些“if else”所提供的稳定性。为了尽可能提高可读性，尽量减少了多线程编程；
- Pipeline 暂时没有时间整理成文档，可以参考 vio.RunOnce() 的流程，以及 frontend.RunOnce() 和 backend.RunOnce()；
