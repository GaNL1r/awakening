# AWAKENING
武汉科技大学RoboMaster崇实战队 视觉算法代码仓库
---
* 持续开发中，欢迎向我们提供本仓库的错误与问题，欢迎进行技术交流
## 写在前面
### 作者
* 刘碧洁 https://github.com/MoCI-L 雷达识别/通信部分开发与维护
* 岳长鑫 https://github.com/TRIAuAuAu 英雄图传/工程机械臂规划（未在本仓库）
* 武晓健 https://github.com/hyheiyue qq:1836871898 vx:hy_xiaojian 自动瞄准/能量机关/哨兵决策/哨兵导航（未在本仓库）/雷达识别/镖体、镖架制导（未在本仓库）
### 对本项目有帮助的RoboMaster开源项目或者个人
* 华南师范大学PIONEER战队@chenjunnn [rm_vision](https://github.com/chenjunnn/rm_vision)
* 中南大学FYT战队 https://github.com/CSU-FYT-Vision/FYT2024_vision
* 河北科技大学Actor&Thinker战队 https://github.com/PraySky1337/at_vision https://github.com/Blackjack200/talos-scheduler https://github.com/Blackjack200/at_vision_simulator
* 同济大学superpower战队 https://github.com/TongjiSuperPower/sp_vision_25
* 深圳北理莫斯科大学北极熊战队 https://github.com/SMBU-PolarBear-Robotics-Team/armor_detector_tensorrt
* 四川大学火锅战队 https://github.com/Ericsii/rm_vision/tree/develop/openvino_armor_detector
* 沈阳航空航天大学TUP战队 https://github.com/tup-robomaster/TUP-Vision-2023-Based/blob/main/src/vehicle_system/autoaim/armor_detector/model/opt-1208-001.onnx
* 北京科技大学Reborn战队 https://github.com/RebornVision/Reborn-Vision-2024-armor-Inference/blob/main/model/number_classifier.onnx
* 华北理工大学Horizon战队 https://github.com/BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025/releases/tag/2026.04.24
* 香港科技大学ENTERPRIZE战队 https://github.com/hkustenterprize/RM2025-Radar-Algorithm
* 南京理工大学Alliance战队 https://github.com/Alliance-Algorithm
* 深圳大学RobotPilots战队 https://github.com/broalantaps/RobotDetectionModel
* 华中科技大学狼牙战队 https://github.com/HUSTLYRM/HUST_HeroAim_2024
* 吉林大学TARS Go战队 https://github.com/Fskaaaaaaaa/jlu_vision_26
* 华南理工大学华南虎战队 https://github.com/scutrobotlab/rm_vision_core https://github.com/cv-rmvl/rmvl
* 哈尔滨工业大学（深圳）南工骁鹰战队 https://github.com/PageChen04/radar_ros_ws
* 东北大学TDT战队 https://github.com/T-DT-Algorithm-2024/T-DT-2024-Radar https://github.com/T-DT-Algorithm-2025/T-DT_Radar
* 福建理工大学仓侠战队 https://github.com/iowqi/PowerRuneSimulator
* 上海交通大学交龙战队 https://github.com/julyfun/rm.cv.fans/blob/main/aimer/base/math/filter/adaptive_ekf.hpp
* 武汉工程大学Nautilus战队 费钰涵
* 江苏开放大学开拓者战队 刘俊豪
* 江苏大学Aurora战队 聂政华
### 前作
https://github.com/WUST-RM/wust_vision
## 亮点
* 集成自动瞄准、能量机关（在老代码，暂时未同步）、全向感知、哨兵决策、雷达识别，方便通用算法与代码复用避免重复造轮子（其实导航的代码也可以放在这里，不过笔者把导航视为了可以真正投入生产生活的算法并不只服务于比赛，所以与本部分鄂进行了区分，也没有放到战队代码仓库），方便统一管理与调试
* 高性能：以自动瞄准算法为例，在intel nuc 1240P平台做到全流程（包括相机取流解码，神经网络推理，cv算法识别，滤波器更新预测等，下同）250hz（工业相机上限）吞吐，平均6-8ms延迟，nvidia orin nx 8g做到 250hz吞吐，4-7ms延迟 （具体实现方案请参考下文内容）
* 无复杂依赖，易部署（一台只配置过操作系统的minipc从0配置只需要不到半小时即可做到上场水平，别问为什么知道）
* 算法实现与应用层分离，可针对不同兵种不同车型不同需求定制不同的运行时代码
## 部署
### 依赖
* llvm21 + cmake + ninja
* OpenCV
* [OpenVINO](https://flowus.cn/7a2a3341-74a1-4db9-bced-99fe5d05ab75)（2024.0.0+）/[TensorRT-cuda](https://flowus.cn/e98af178-de0b-4546-808d-a6f1ff199d62)(TensorRT 10.6.0.26 cuda 12-6)
* fmt
* ceres
* Eigen3
* nlohmann
* yaml-cpp
* spdlog
* HikSDK/MvSDK
* ROS2（可选）
### quick-start
```
git clone --recurse-submodules https://github.com/WUST-RM/awakening.git
cd awakening
sudo ./run run/debug/race/build/rebuild args... # run:编译且运行 第一个参数为bin中exe名 其余为exe参数（详细请见代码实现） debug 以gdb形式运行 参数规则同run race:不编译直接运行 参数规则同run build:仅编译 rebuild:清除缓存全量重编译 注意：修改cmakelists中的option需要rebuild清除缓存才能生效
# eg: sudo ./run.sh run auto_aim config/omni.yaml true
python3 web.py #运行远程可视化web（雷达识别不适用）
```
## 自动瞄准/能量机关

