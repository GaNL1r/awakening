# AWAKENING

武汉科技大学RoboMaster崇实战队 视觉算法代码仓库

---

- 持续开发中，欢迎向我们提供本仓库的错误与问题，欢迎进行技术交流

## 写在前面

### 作者

- 刘璧洁 [MoCI-L](https://github.com/MoCI-L) 雷达识别/通信部分开发与维护
- 岳长鑫 [TRIAuAuAu](https://github.com/TRIAuAuAu) 英雄图传/工程机械臂规划（未在本仓库）
- 武晓健 [hyheiyue](https://github.com/hyheiyue) qq:1836871898 vx:hy_xiaojian 自动瞄准/能量机关/哨兵决策/哨兵导航（未在本仓库）/雷达识别/镖体、镖架制导（未在本仓库）

### 对本项目有帮助的RoboMaster开源项目或者个人（排名不分先后）

- 华南师范大学PIONEER战队@chenjunnn [rm_vision](https://github.com/chenjunnn/rm_vision)
- 中南大学FYT战队 [FYT2024_vision](https://github.com/CSU-FYT-Vision/FYT2024_vision)
- 河北科技大学Actor&Thinker战队 [at_vision](https://github.com/PraySky1337/at_vision) [talos-scheduler](https://github.com/Blackjack200/talos-scheduler)  [at_vision_simulator](https://github.com/Blackjack200/at_vision_simulator)
- 同济大学superpower战队 [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
- 深圳北理莫斯科大学北极熊战队 [armor_detector_tensorrt](https://github.com/SMBU-PolarBear-Robotics-Team/armor_detector_tensorrt)
- 四川大学火锅战队 [openvino_armor_detector](https://github.com/Ericsii/rm_vision/tree/develop/openvino_armor_detector)
- 沈阳航空航天大学TUP战队 [opt-1208-001.onnx](https://github.com/tup-robomaster/TUP-Vision-2023-Based/blob/main/src/vehicle_system/autoaim/armor_detector/model/opt-1208-001.onnx)
- 北京科技大学Reborn战队 [number_classifier.onnx](https://github.com/RebornVision/Reborn-Vision-2024-armor-Inference/blob/main/model/number_classifier.onnx)
- 华北理工大学Horizon战队 https://github.com/BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025/releases/tag/2026.04.24
- 香港科技大学ENTERPRIZE战队 [RM2025-Radar-Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm)
- 南京理工大学Alliance战队 https://github.com/Alliance-Algorithm
- 五大湖联合大学The Great Lakes 战队 [Pacific_doorlock_sniper](https://github.com/wele0612/Pacific_doorlock_sniper)
- 深圳大学RobotPilots战队 [RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel)
- 华中科技大学狼牙战队 [HUST_HeroAim_2024](https://github.com/HUSTLYRM/HUST_HeroAim_2024)
- 吉林大学TARS Go战队 [jlu_vision_26](https://github.com/Fskaaaaaaaa/jlu_vision_26)
- 华南理工大学华南虎战队 [rm_vision_core](https://github.com/scutrobotlab/rm_vision_core) [rmvl](https://github.com/cv-rmvl/rmvl)
- 哈尔滨工业大学（深圳）南工骁鹰战队 [radar_ros_ws](https://github.com/PageChen04/radar_ros_ws)
- 东北大学TDT战队 [T-DT-2024-Radar](https://github.com/T-DT-Algorithm-2024/T-DT-2024-Radar) [T-DT_Radar](https://github.com/T-DT-Algorithm-2025/T-DT_Radar)
- 福建理工大学仓侠战队 [PowerRuneSimulator](https://github.com/iowqi/PowerRuneSimulator)
- 上海交通大学交龙战队 [adaptive_ekf.hpp](https://github.com/julyfun/rm.cv.fans/blob/main/aimer/base/math/filter/adaptive_ekf.hpp)
- 武汉工程大学Nautilus战队 费钰涵
- 江苏开放大学开拓者战队 刘俊豪
- 江苏大学Aurora战队 聂政华

### 前作

[wust_vision](https://github.com/WUST-RM/wust_vision)

## 亮点

- 集成自动瞄准、能量机关（在老代码，暂时未同步）、全向感知、哨兵决策、雷达识别，方便了通用算法与代码复用避免重复造轮子（其实导航的代码也可以放在这里，不过笔者把导航视为了可以真正投入生产生活的算法并不只服务于比赛，所以与本部分进行了区分，也没有放到战队代码仓库），方便统一管理与调试
- 高性能：以自动瞄准算法为例，在intel nuc 1240P平台做到全流程（包括相机取流解码，神经网络推理，cv算法识别，滤波器更新预测等，下同）250hz（工业相机上限）吞吐，平均6-8ms延迟，nvidia orin nx 8g做到 250hz吞吐，4-7ms延迟 （具体实现方案请参考下文内容）
- 无复杂依赖，易部署（一台只配置过操作系统的minipc从0配置只需要不到半小时即可做到上场水平，别问为什么知道）
- 算法实现与应用层分离，可针对不同兵种不同车型不同需求定制不同的运行时代码

## 部署

### 依赖

- llvm21 + cmake + ninja
- OpenCV
- [OpenVINO](https://flowus.cn/7a2a3341-74a1-4db9-bced-99fe5d05ab75)（2024.0.0+）/[TensorRT-cuda](https://flowus.cn/e98af178-de0b-4546-808d-a6f1ff199d62)(TensorRT 10.6.0.26 cuda 12-6)
- fmt
- ceres
- Eigen3
- nlohmann
- yaml-cpp
- spdlog
- HikSDK/MvSDK
- ROS2（可选）

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

- 单帧有效观测偏少。典型做法是只跟踪当前最可信的一块装甲板，在装甲板切换或目标旋转时，再通过类似 `handle_armor_jump` 的逻辑修改状态量来完成观测目标切换。这种方法工程上可行，但单帧中其他可见装甲板、半遮挡灯条等信息没有被充分利用。
- 很多方案以 PnP 解出的装甲板位姿作为主要观测。PnP 本身依赖角点质量、装甲板尺寸、视角和距离，远距离小目标、斜视角、共面矩形目标都会让深度和 yaw 抖动变大。即使后续使用 EKF，滤波器看到的也已经是 PnP 压缩后的三维结果，而不是原始图像测量。
- 单独灯条难以被传统 PnP 链路利用。单灯条只有上下两个关键点，不足以独立求解完整 6DoF 位姿，但它仍然携带目标位置、朝向和尺度信息。如果算法只能处理完整装甲板，就会在遮挡、远距离和侧身场景中损失大量可用观测。

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

### 基于“整车”状态先验的显式空间注意力机制

大多数RoboMaster 视觉算法使用的神经网络模型都采用固定尺寸输入，需要在输入神经网络前通常需要对工业相机图像进行缩放、填充或裁剪。常见做法是使用 `letterbox`：在保持原图长宽比的前提下，将图像缩放到网络输入尺寸，再用 `padding` 补齐空白区域。该方法能避免几何形变，但也带来一个问题：工业相机图像的长宽比通常与网络输入长宽比不一致，整幅图经过 `letterbox` 后往往会被整体缩小，远距离装甲板本来只占很少像素，缩放后有效纹理和边缘信息进一步减少，导致关键点回归、分类和置信度都会下降。

我们利用“整车”状态估计提供的几何先验，在检测阶段引入一种显式的空间注意力机制。不再把整张图像无差别地送入检测器，而是根据当前整车状态、相机外参和时间戳预测，将目标在当前图像中的可能区域重投影出来，并构造与神经网络输入比例一致的 ROI。这样网络输入关注的是目标高概率出现区域，而不是包含大量背景的整幅图像。

对于神经网络识别链路，这种 ROI 机制有两个直接收益。第一，ROI 的长宽比可以主动匹配网络输入，减少 letterbox padding 和无效背景区域，使输入像素更多用于描述目标本身。第二，当目标距离较远时，ROI 裁剪后再 resize 到网络输入尺寸，相当于对目标区域进行局部放大，保留并增强远距离小目标的灯条边缘、数字区域和角点结构。相比全图缩放，网络看到的目标像素占比更高，关键点回归更稳定，远距离小目标的检出率和回归质量都会明显提升。

对于传统 CV 算法识别链路，同样可以利用整车先验构造更小的 ROI。传统灯条检测通常依赖颜色阈值、亮度分割、轮廓提取、形态筛选和灯条配对，搜索区域越大，环境灯光等干扰越容易进入候选集，同时 CPU 开销和误匹配概率也会增加。根据整车状态直接预测各装甲板灯条端点在图像中的位置，生成紧凑的 ROI，使传统算法只处理目标附近区域，从源头减少无效候选，降低CPU 压力，并提升处理速度和匹配稳定性。

因此，本项目的检测前处理可以概括为：利用“整车”状态估计提供的时空先验，在图像平面上动态生成网络 ROI 和传统 CV ROI。网络 ROI 更关注比例匹配和小目标细节保留，传统 CV ROI 更关注搜索范围收缩和干扰抑制。相比常见的全图 letterbox 检测或基于上一帧 2D 框的简单 ROI，本方法把运动预测、相机投影和整车几何约束结合起来，形成了一种更稳定、更可解释的显式空间注意力机制，在远距离、目标旋转等复杂背景场景下具有更好的工程收益。

### 类 MINCO 思想的开环轨迹规划  

同济大学 SuperPower 战队在 `sp_vision_25` 中提出了“轨迹视角下的自瞄理论”：自瞄效果不应只看某一时刻的瞄准点是否正确，而应看云台轨迹与射击轨迹在一段时间内的重合度。云台轨迹越接近考虑弹丸飞行时间后的射击轨迹，单位时间内可开火窗口越大，理论 DPS 越高，击杀时间越短。基于这一观点，轨迹规划器的目标可以表述为：

```text
在云台最大加速度约束内，使规划后的控制轨迹尽可能贴近射击轨迹。
```

`sp_vision_25` 给出了两类实现思路。第一类是隐式搜索：将一段时间内的控制轨迹离散采样，以轨迹重合度作为优化目标，以最大加速度作为约束，将问题转化为二次规划并使用 TinyMPC 求解。该方案表达能力强，也能获得较好的规划效果，但优化器本身相对“重”：需要维护优化问题、矩阵形式、求解器参数和迭代过程，计算流程更像黑箱。

第二类是显式搜索：以过渡时间为变量，确定切板点前后的起点和终点，用五次多项式生成过渡段轨迹，并逐步调整过渡时间，直到满足最大加速度限制。该方法的思想非常直接：不去优化整段射击轨迹，只在不连续切板附近插入一段可执行的平滑过渡；远离切板的跟随段保持与射击轨迹重合。这样既保留了大部分可开火时间，又避免在切换点要求云台完成物理上不可实现的瞬时跳变。

本项目在这一思路上进一步做了轻量化实现，并引入了类似 `MINCO` 的参数化思想。MINCO 在移动机器人轨迹规划中的核心价值，是将多项式系数从优化变量中解析消去，使优化器主要面对中间点和时间分配。本项目的 `LimitTrajectory` 也采用类似思想：控制点位置不作为优化变量，五次多项式系数由边界状态闭式求解，在线规划只调整切板附近的时间覆盖范围。

具体来说，系统首先在当前时刻前后采样一段目标射击轨迹。每个采样点都由整车状态预测、弹道解算和装甲板选择共同确定，得到对应的云台 `yaw`、`pitch` 控制点。随后规划器对这些控制点估计节点速度和加速度，并在相邻节点之间构造五次多项式段：

```text
p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
```

五次多项式的优势在于，它可以同时满足起点和终点的位置、速度、加速度约束：

```text
p(0), v(0), a(0), p(T), v(T), a(T)
```

因此每一段轨迹天然具有位置、速度和加速度连续性。实现上不需要求解通用优化问题，只需要根据边界条件直接计算 `c0` 到 `c5`。这使得轨迹生成过程稳定、确定、计算量小，也避免了在线 QP 或 MPC 求解器带来的额外依赖。

难点集中在切板点。切换装甲板时，目标控制点会发生突变，若仍使用原始采样间隔生成多项式，过渡段会得到超过云台能力的加速度。我们提出的 `LimitTrajectory` 会先找到当前时间附近最近的切板区间，然后以该区间为中心，不改变控制点位置，只向左右扩大过渡段覆盖的时间范围。时间越长，同样的位置变化被分摊到更长时间内完成，所需加速度越低。规划器不断扩大该过渡段，直到解析计算得到的最大加速度满足配置约束：

```text
max |a_yaw(t)|   <= max_yaw_acc
max |a_pitch(t)| <= max_pitch_acc
```

这里的最大加速度不是靠密集采样粗略估计，而是由多项式解析计算。对于五次位置多项式，其加速度是三次函数，jerk 是二次函数。加速度极值只可能出现在区间端点或 jerk 为零的内部点。实现中通过求解 jerk 的根来检查这些候选点，从而得到该段轨迹的最大绝对加速度。这比固定步长扫描更精确，也更适合高频在线规划。

这种方法可以理解为一种受限参数空间内的显式最优搜索：在“控制点固定、只调整切板过渡时间”的假设下，规划器寻找满足加速度约束的最短过渡段。过渡段越短，轨迹偏离原始射击轨迹的时间越少；过渡段越长，云台执行越平滑但可开火窗口会下降。因此，该规划器实际优化的是“可执行性”和“射击轨迹重合度”之间的折中。

从控制角度看，这是一种开环前馈轨迹规划器。它不直接使用云台实际反馈参与在线闭环优化，而是根据目标预测轨迹提前生成未来一段时间的 `yaw`、`pitch`、速度和加速度指令。下位机控制器可以将这些速度、加速度作为前馈量，与自身闭环误差控制叠加，从而降低切板时的控制压力。

从开火决策角度看，规划后的控制轨迹和原始射击轨迹可以同时被查询。系统不仅能判断当前云台是否位于可命中窗口内，还能考虑发射延迟，在未来若干时刻检查控制轨迹与射击轨迹的偏差。如果延迟区间内轨迹误差会超过装甲板可命中范围，则提前禁止开火。这使开火逻辑从经验阈值判断转向轨迹一致性判断。

因此，本项目的 `LimitTrajectory` 并不是MPC控制器，而是借鉴了 `MINCO` “固定几何点、解析消元多项式系数、主要搜索时间分配”的思想，将其改造成适合自动瞄准切板问题的轻量级开环轨迹规划器。用很低的计算成本解决了装甲板切换带来的轨迹不连续问题，在保证最大加速度约束的同时，尽可能保持云台控制轨迹与射击轨迹重合，从而提升高转速目标下的跟随稳定性和有效开火窗口。
