# AWAKENING

武汉科技大学RoboMaster崇实战队 视觉算法代码仓库

---

* 持续开发中，欢迎向我们提供本仓库的错误与问题，欢迎进行技术交流

## 写在前面

### 作者

* 刘碧洁 https://github.com/MoCI-L 雷达识别/通信部分开发与维护
* 岳长鑫 https://github.com/TRIAuAuAu 英雄图传/工程机械臂规划（未在本仓库）
* 武晓健 https://github.com/hyheiyue qq:1836871898 vx:hy_xiaojian 自动瞄准/能量机关/哨兵决策/哨兵导航（未在本仓库）/雷达识别/镖体、镖架制导（未在本仓库）

### 对本项目有帮助的RoboMaster开源项目或者个人（排名不分先后）

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
* 五大湖联合大学The Great Lakes 战队 https://github.com/wele0612/Pacific_doorlock_sniper
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

* 集成自动瞄准、能量机关（在老代码，暂时未同步）、全向感知、哨兵决策、雷达识别，方便了通用算法与代码复用避免重复造轮子（其实导航的代码也可以放在这里，不过笔者把导航视为了可以真正投入生产生活的算法并不只服务于比赛，所以与本部分进行了区分，也没有放到战队代码仓库），方便统一管理与调试
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

```bash
git clone --recurse-submodules https://github.com/WUST-RM/awakening.git
cd awakening
sudo ./run run/debug/race/build/rebuild args... # run:编译且运行 第一个参数为bin中exe名 其余为exe参数（详细请见代码实现） debug 以gdb形式运行 参数规则同run race:不编译直接运行 参数规则同run build:仅编译 rebuild:清除缓存全量重编译 注意：修改cmakelists中的option需要rebuild清除缓存才能生效
# eg: sudo ./run.sh run auto_aim config/omni.yaml true
python3 web.py #运行远程可视化web（雷达识别不适用）
```

## 自动瞄准/能量机关

### 高效多观测“整车”状态估计

由于机器人制作规范对于机器人设计的限制，机器人装甲板具有明显的几何排列特征：同一辆车上的多块装甲板并不是互相独立的目标，而是固定在同一个刚体上的多个观测面。能量机关也具有类似特点，靶盘、中心 R 标和叶片之间存在稳定的机械约束。也正因为如此，在 RoboMaster 视觉算法中，“整车”状态估计一直是一个重要方向。

以往不少开源方案已经引入了整车模型，但在实际使用中仍常见几个问题：

---

* 单帧有效观测偏少。典型做法是只跟踪当前最可信的一块装甲板，在装甲板切换或目标旋转时，再通过类似 `handle_armor_jump` 的逻辑修改状态量来完成观测目标切换。这种方法工程上可行，但单帧中其他可见装甲板、半遮挡灯条等信息没有被充分利用。
* 很多方案以 PnP 解出的装甲板位姿作为主要观测。PnP 本身依赖角点质量、装甲板尺寸、视角和距离，远距离小目标、斜视角、共面矩形目标都会让深度和 yaw 抖动变大。即使后续使用 EKF，滤波器看到的也已经是 PnP 压缩后的三维结果，而不是原始图像测量。
* 单独灯条难以被传统 PnP 链路利用。单灯条只有上下两个关键点，不足以独立求解完整 6DoF 位姿，但它仍然携带目标位置、朝向和尺度信息。如果算法只能处理完整装甲板，就会在遮挡、远距离和侧身场景中损失大量可用观测。

---

针对这些痛点，我们设计了一个基于图像点重投影的多观测“整车”状态估计算法。它不把每块装甲板当作彼此独立的目标，而是先建立整车状态，再根据整车几何预测每块装甲板和灯条在图像中的位置，最后用实际检测到的图像点反向更新整车状态。

算法核心仍然是扩展卡尔曼滤波器，但观测方式和传统 `PnP -> EKF` 链路不同：PnP 只在初始化和粗匹配阶段提供初值，真正的状态更新发生在图像平面。滤波器在线性化过程中把当前整车状态重投影到图像上，与装甲板角点、灯条端点等观测形成像素残差，再通过多观测更新得到更一致的整车状态。

以自动瞄准部分的 4 装甲板“整车”状态估计为例，本项目维护的状态可以理解为：

```text
x = [cx, vcx, cy, vcy, cz, vcz, yaw, vyaw, r, l, h]^T
```

其中 `c = (cx, cy, cz)` 表示整车中心位置，`v = (vcx, vcy, vcz)` 表示整车速度，`yaw` 和 `vyaw` 表示整车朝向及角速度，`r` 表示装甲板到车体中心的基础半径，`l` 和 `h` 用来描述四装甲板车辆长短轴或高度差带来的几何差异。

对于第 `i` 块装甲板，其相对车体中心的位置由整车状态直接给出：

```text
theta_i = yaw + i * 2pi / 4
r_i = r 或 r + l
p_i = [cx, cy, cz]^T + [-r_i cos(theta_i), -r_i sin(theta_i), dz_i]^T
```

这一步的意义是把“装甲板跳变”从经验补丁变成几何关系。无论当前看到的是正面装甲板、侧面装甲板，还是相邻装甲板的一条灯条，它们本质上都是同一个整车状态在不同位置上的投影。只要观测能匹配到对应的装甲板编号，就可以共同约束同一个状态向量。

观测模型则直接工作在图像平面。对某块装甲板或某条灯条，算法根据当前状态生成其三维位姿，再通过相机模型投影到像素坐标：

```text
z_hat = project(camera, T_camera_odom^-1 * T_armor_i(x) * P)
residual = z_observed - z_hat
```

这里 `z_observed` 是检测器给出的真实角点或灯条端点，`z_hat` 是当前整车状态预测出的像素位置。滤波器最小化的不是 PnP 后的三维位姿误差，而是更接近传感器原始测量的像素重投影误差。

从数学上看，多观测更新可以写成：

```text
z = [z_1, z_2, ..., z_n]^T
h(x) = [h_1(x), h_2(x), ..., h_n(x)]^T
residual = z - h(x)
```

扩展卡尔曼滤波器在当前状态附近对 `h(x)` 线性化，得到观测雅可比 `H`，再根据观测噪声 `R` 和状态协方差 `P` 计算卡尔曼增益：

```text
K = P H^T (H P H^T + R)^-1
x = x + K (z - h(x))
```

当一帧中加入更多有效观测时，`H^T R^-1 H` 提供的信息量增加，后验状态的不确定性会下降。也就是说，“同时观测装甲板和灯条”不是简单堆特征点，而是在滤波理论上增加了对整车状态的约束。

在工程实现上，本项目还使用门控贪心匹配来避免错误观测进入滤波器。对于装甲板，简单的图像位置误差维度过低，这里使用简单 PnP 得到 yaw、pitch、distance 与旋转 yaw，并构造马氏距离进行匹配；对于灯条，对长度、倾角和距离进行门限，然后使用距离作为代价匹配。只有通过门控的装甲板和灯条才会构造观测项。这样既能充分利用多观测，又能降低误匹配对状态估计的破坏。

因此，本项目的“整车”状态估计并不是传统意义上“先 PnP 再滤波”的简单改写，而是把整车几何约束前移到观测模型中：由整车状态直接生成装甲板和灯条的图像预测，再用真实图像点反向修正整车状态。它在理论上更接近“带刚体几何约束的多特征视觉状态估计”，在比赛场景中则体现为更高的观测利用率、更稳定的切板过程，以及更好的遮挡和远距离鲁棒性。
