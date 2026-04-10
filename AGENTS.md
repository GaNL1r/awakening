# AGENTS.md

## Build Commands

```bash
./run.sh build           # Incremental build (creates build/ dir)
./run.sh rebuild         # Clean rebuild (deletes build/ first)
./run.sh run auto_aim    # Build and run main executable
```

Uses CMake + Ninja with clang/clang++ (not gcc). Output binaries go to `bin/`.

## Environment Setup

Before building/running, source environment:
```bash
source env.bash
```

Sets HIK camera SDK paths and `VISION_ROOT`. The `run.sh` script sources this automatically.

## Architecture

- **Main executable**: `auto_aim` (src/runtime/auto_aim.cpp)
- **Libraries**: `awakening_utils`, `awakening_auto_aim`, optional `awakening_rcl` (ROS2)
- **Config**: YAML files in `config/`, use `${ROOT_DIR}` token for repo-relative paths
- **3rdparty**: KalmanHyLib, backward-cpp (stack traces), ankerl (hash map)

## Build Options

CMake options (default ON):
- `BUILD_WITH_TRT` - TensorRT backend (requires CUDA)
- `BUILD_WITH_OPENVINO` - OpenVINO backend

ROS2 integration compiles automatically if all ROS2 packages are found.

## Code Style

Run clang-format before committing:
```bash
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cu' -o -name '*.cpp' \) -exec clang-format -i {} +
```

Uses custom `.clang-format` config (4-space indent, column limit 100).

## Debug Interface

Web dashboard (web.py) runs on port 8000, reads debug data from:
- `/dev/shm/awakening_frame` - MJPEG frames
- `/dev/shm/awakening_data.json` - Runtime data
- `/dev/shm/awakening_log.json` - Logs

## Notes

- Config tokens like `${ROOT_DIR}` in YAML are replaced at runtime via `param_deliver.h`
- `.clangd` configures language server for C++23 and CUDA includes
- No test framework configured
- 回答时请使用中文

## 项目架构详细分析

### 整体架构

这是一个用于RoboMaster比赛的自瞄系统（Auto-Aim System），采用模块化、基于数据流的架构设计。系统通过任务调度器（Scheduler）管理多个并行任务节点，实现图像采集、装甲板检测、目标跟踪、弹道解算等功能。

### 核心组件层次结构

```
awakening (项目根)
├── src/
│   ├── runtime/                    # 运行时入口
│   │   └── auto_aim.cpp           # 主程序入口（main函数）
│   ├── utils/                      # 工具库 (awakening_utils)
│   │   ├── scheduler/             # 任务调度系统
│   │   │   ├── scheduler.hpp      # 调度器核心
│   │   │   └── node.hpp           # 任务节点定义
│   │   ├── drivers/               # 硬件驱动
│   │   │   ├── hik_camera.hpp     # 海康相机驱动
│   │   │   └── serial_driver.hpp  # 串口驱动
│   │   ├── net_detector/          # 神经网络推理
│   │   │   ├── net_detector_base.hpp       # 推理接口基类
│   │   │   ├── tensorrt/          # TensorRT后端
│   │   │   └── openvino/          # OpenVINO后端
│   │   ├── tf.hpp                 # 坐标变换系统
│   │   ├── buffer.hpp             # 线程安全缓冲区
│   │   └── logger.hpp             # 日志系统
│   ├── tasks/                      # 任务模块
│   │   ├── auto_aim/              # 自瞄核心模块 (awakening_auto_aim)
│   │   │   ├── armor_detect/      # 装甲板检测
│   │   │   │   ├── armor_detector.hpp
│   │   │   │   └── armor_infer.hpp
│   │   │   ├── armor_tracker/     # 目标跟踪
│   │   │   │   ├── armor_tracker.hpp
│   │   │   │   ├── armor_target.hpp
│   │   │   │   ├── motion_model.hpp      # 运动模型
│   │   │   │   └── motion_model_point.hpp
│   │   │   ├── armor_control/     # 控制解算
│   │   │   │   └── very_aimer.hpp
│   │   │   ├── auto_aim_fsm.hpp   # 状态机
│   │   │   ├── debug.hpp          # 调试可视化
│   │   │   └── type.hpp           # 数据类型定义
│   │   └── base/                  # 基础组件
│   │       ├── common.hpp         # 通用数据结构
│   │       ├── ballistic_trajectory.hpp  # 弹道模型
│   │       ├── web.hpp            # Web调试接口
│   │       ├── recorder_player..hpp      # 录制/回放
│   │       └── packet_typedef.hpp # 通信协议
│   └── _rcl/                      # ROS2集成 (可选)
│       ├── node.hpp               # ROS2节点
│       ├── tf.hpp                 # ROS2 TF
│       └── visual/                # 可视化
├── config/                        # 配置文件
│   ├── test.yaml                  # 测试配置
│   └── omni.yaml                  # 全向配置
├── model/                         # 模型文件
└── web.py                         # Web调试服务器
```

### 运行时调用关系

#### 1. 启动流程 (main函数)

```
main()
  ├─ 初始化信号处理 (SignalGuard)
  ├─ 初始化日志系统 (logger::init)
  ├─ 解析命令行参数
  │   ├─ argv[1]: 配置文件路径
  │   └─ argv[2]: 是否启用调试模式
  ├─ 加载YAML配置
  ├─ 初始化录制器/播放器 (可选)
  ├─ 创建任务调度器 (Scheduler)
  └─ 初始化硬件驱动
      ├─ 串口驱动 (SerialDriver) - 接收机器人姿态数据
      └─ 海康相机 (HikCamera) - 采集图像帧
```

#### 2. 核心数据流图

```
┌─────────────┐
│  数据源层    │
└──────┬──────┘
       │
       ├─ Camera (HikCamera)
       │   └─ 输出: ImageFrame [时间戳 + RGB图像]
       │
       └─ Serial (SerialDriver)
           └─ 输出: ReceiveRobotData [yaw/pitch/roll/弹数]
              └─ 更新 TF 树: GIMBAL_ODOM → GIMBAL

┌─────────────┐
│  预处理层    │
└──────┬──────┘
       │
       ├─ push_common_frame
       │   ├─ 输入: ImageFrame
       │   ├─ 处理: 根据跟踪目标预测ROI区域
       │   └─ 输出: CommonFrame [图像 + 时间戳 + 帧ID + ROI]
       │
       ├─ auto_exposure
       │   ├─ 输入: CommonFrame
       │   └─ 处理: 自动曝光调节
       │
       └─ receive_serial
           ├─ 输入: ReceiveRobotData
           ├─ 处理: 更新TF树，记录子弹发射
           └─ 输出: 更新 SimpleRobotTF

┌─────────────┐
│  检测层      │
└──────┬──────┘
       │
       └─ detector
           ├─ 输入: CommonFrame
           ├─ 处理:
           │   ├─ ArmorDetector::detect()
           │   │   ├─ NetDetector (TensorRT/OpenVINO)
           │   │   ├─ 神经网络推理检测装甲板
           │   │   ├─ 关键点提取
           │   │   └─ 颜色/数字分类
           │   └─ 信号量控制并发推理数
           └─ 输出: Armors [装甲板列表 + 时间戳 + 帧ID]

┌─────────────┐
│  跟踪层      │
└──────┬──────┘
       │
       └─ tracker
           ├─ 输入: Armors
           ├─ 处理:
           │   ├─ 敌方颜色过滤
           │   ├─ 坐标变换 (CAMERA_CV → ODOM)
           │   ├─ ArmorTracker::track()
           │   │   ├─ 数据关联（匹配检测与跟踪）
           │   │   ├─ ESEKF状态估计
           │   │   │   ├─ 位置 (cx, cy, cz)
           │   │   │   ├─ 速度 (vx, vy, vz)
           │   │   │   ├─ 姿态 (yaw, vyaw)
           │   │   │   └─ 几何 (r, l, h)
           │   │   └─ 跟踪状态管理
           │   │       ├─ LOST → DETECTING → TRACKING
           │   │       └─ TRACKING ↔ TEMP_LOST
           │   └─ AutoAimFsmController::update()
           │       └─ FSM状态切换
           │           ├─ AIM_SINGLE_ARMOR (单板)
           │           ├─ AIM_WHOLE_CAR_ARMOR (整车)
           │           ├─ AIM_WHOLE_CAR_PAIR (双板)
           │           └─ AIM_WHOLE_CAR_CENTER (中心)
           └─ 输出: ArmorTarget [跟踪目标 + 状态]

┌─────────────┐
│  解算层      │
└──────┬──────┘
       │
       └─ solver (周期任务 1000Hz)
           ├─ 输入: ArmorTarget (共享内存读取)
           ├─ 处理:
           │   ├─ 坐标变换 (目标帧 → GIMBAL_ODOM)
           │   ├─ VeryAimer::very_aim()
           │   │   ├─ 根据FSM状态选择瞄准策略
           │   │   ├─ 弹道补偿
           │   │   │   ├─ 重力补偿
           │   │   │   ├─ 空气阻力
           │   │   │   └─ 迭代求解
           │   │   ├─ 运动预测
           │   │   │   ├─ 位置预测
           │   │   │   └─ 姿态预测
           │   │   └─ 计算云台控制量
           │   │       ├─ yaw/pitch角度
           │   │       ├─ 角速度前馈
           │   │       └─ 开火建议
           │   └─ 通过串口发送 SendRobotCmdData
           └─ 输出: GimbalCmd

┌─────────────┐
│  调试层      │
└──────┬──────┘
       │
       ├─ debug (周期任务 60Hz)
       │   ├─ 读取 ArmorTarget
       │   ├─ 坐标变换到相机坐标系
       │   ├─ 绘制调试信息
       │   │   ├─ 装甲板边框
       │   │   ├─ 瞄准点
       │   │   ├─ 弹道轨迹
       │   │   └─ 状态文本
       │   └─ 写入共享内存
       │       ├─ /dev/shm/awakening_frame (图像)
       │       ├─ /dev/shm/awakening_data.json (数据)
       │       └─ /dev/shm/awakening_log.json (日志)
       │
       └─ logger (周期任务 1Hz)
           └─ 输出统计信息
               ├─ 检测次数
               ├─ 跟踪次数
               ├─ 找到次数
               ├─ 平均延迟
               └─ 串口/相机帧数
```

#### 3. 关键类详解

**Scheduler (任务调度器)**
- 位置: `src/utils/scheduler/scheduler.hpp`
- 功能:
  - 基于任务图的并行调度
  - 自动拓扑连接（通过类型索引匹配输入输出）
  - 支持数据源节点（SourceNode）和任务节点（TaskNode）
  - 内置线程池（替代TBB）
  - 支持周期性任务（RateSource）

**ArmorDetector (装甲板检测器)**
- 位置: `src/tasks/auto_aim/armor_detect/armor_detector.hpp`
- 功能:
  - 神经网络推理（支持TensorRT/OpenVINO后端）
  - 装甲板关键点检测（4个角点）
  - 颜色分类（红/蓝/紫）
  - 数字识别（0-5号/哨兵/基地/前哨站）

**ArmorTracker (目标跟踪器)**
- 位置: `src/tasks/auto_aim/armor_tracker/armor_tracker.hpp`
- 功能:
  - 基于ESEKF的状态估计
  - 数据关联（匹配检测与跟踪目标）
  - 跟踪生命周期管理
  - 运动建模（匀速旋转模型）

**ArmorTarget (跟踪目标)**
- 位置: `src/tasks/auto_aim/armor_tracker/armor_target.hpp`
- 核心状态向量（12维）:
  - 位置: cx, cy, cz (中心坐标)
  - 速度: vx, vy, vz (线速度)
  - 姿态: yaw (偏航角), vyaw (角速度)
  - 几何: r (半径), l (装甲板间距), h (高度), dz (z偏移)

**VeryAimer (瞄准解算器)**
- 位置: `src/tasks/auto_aim/armor_control/very_aimer.hpp`
- 功能:
  - 根据FSM状态选择瞄准策略
  - 弹道解算（考虑重力、空气阻力）
  - 运动预测（位置、姿态）
  - 输出云台控制指令

**AutoAimFsmController (状态机)**
- 位置: `src/tasks/auto_aim/auto_aim_fsm.hpp`
- 状态转换:
  ```
  AIM_SINGLE_ARMOR → AIM_WHOLE_CAR_ARMOR → AIM_WHOLE_CAR_PAIR → AIM_WHOLE_CAR_CENTER
         ↑                                      ↓                      ↓
         └──────────────────────────────────────┴──────────────────────┘
  ```
  - 根据目标旋转速度（vyaw）动态切换瞄准策略

#### 4. 坐标系与变换

**坐标系定义** (SimpleFrame枚举):
```
ODOM (世界坐标系)
  └─ GIMBAL_ODOM (云台里程计)
      └─ GIMBAL (云台坐标系)
          ├─ CAMERA (相机坐标系)
          │   └─ CAMERA_CV (OpenCV图像坐标系)
          └─ SHOOT (发射坐标系)
```

**关键变换**:
- `camera_in_gimbal`: 相机相对云台的安装位姿
- `shoot_in_gimbal`: 发射点相对云台的位姿
- `gimbal_odom_in_odom`: 云台里程计相对世界坐标（由串口数据更新）

**坐标变换系统** (SimpleRobotTF):
- 基于图结构的多坐标系变换
- 支持时间戳查询（历史变换插值）
- 自动查找最短路径变换链

#### 5. 数据类型

**核心数据结构**:
- `ImageFrame`: 图像帧（时间戳 + RGB数据）
- `CommonFrame`: 处理帧（图像 + ROI + 帧ID）
- `Armor`: 装甲板检测结果（颜色/类别/关键点/位姿）
- `Armors`: 装甲板列表（时间戳 + 帧ID）
- `ArmorTarget`: 跟踪目标（状态 + 跟踪状态）
- `GimbalCmd`: 云台控制指令（角度/角速度/开火建议）

**枚举类型**:
- `EnemyColor`: 敌方颜色（RED/BLUE）
- `ArmorColor`: 装甲板颜色（BLUE/RED/NONE/PURPLE）
- `ArmorClass`: 装甲板类别（SENTRY/NO1-5/OUTPOST/BASE）
- `ArmorType`: 装甲板类型（SimpleSmall/Large）
- `AutoAimFsm`: FSM状态

#### 6. 配置文件结构

```yaml
enemy_color: red                    # 敌方颜色
bullet_speed: 23.0                  # 弹速
max_infer_num: 5                    # 最大并发推理数

serial:                             # 串口配置
  enable: true
  device_name: /dev/ttyACM0
  baud_rate: 115200

tf:                                 # 坐标变换
  camera_in_gimbal: {t: [...], R: [...]}
  shoot_in_gimbal: {t: [...], R: [...]}

armor_detector:                     # 检测器配置
  armor_infer:
    model_type: tup
    conf_threshold: 0.2
  net_detector:
    backend: tensorrt

armor_tracker:                      # 跟踪器配置
  esekf_iter_num: 3
  lost_time_thres: 0.5
  tracking_thres: 5
  match_gate: 2.0

auto_aim_fsm:                       # FSM配置
  transfer_thresh: 20
  single_whole_up: 2.5

very_aimer:                         # 解算器配置
  fly_time_thres: 0.4
  shoot_delay: 0.02

bullet_pick_up:                     # 弹道配置
  ballistic_trajectory:
    gravity: 9.8
    resistance: 0.092
```

#### 7. 调试与可视化

**Web调试界面** (`web.py`):
- 运行: `python web.py`
- 端口: 8000
- 数据源:
  - `/dev/shm/awakening_frame`: MJPEG视频流
  - `/dev/shm/awakening_data.json`: 实时数据
  - `/dev/shm/awakening_log.json`: 日志

**调试模式** (`debug=true`):
- 绘制装甲板边框
- 显示瞄准点与弹道
- 输出FSM状态
- 记录平均延迟

**ROS2集成** (可选):
- 发布TF变换
- 发布可视化marker
- 节点名称: `auto_aim`

#### 8. 性能优化

- **并发推理**: 通过信号量限制最大并发数（`max_infer_num`）
- **ROI裁剪**: 根据跟踪结果预测下一帧目标区域
- **线程池**: 替代TBB，自定义线程池管理
- **零拷贝**: 使用移动语义减少数据拷贝
- **批量处理**: 支持批量装甲板检测输出

### 关键设计模式

1. **任务图模式**: Scheduler通过类型索引自动连接任务节点
2. **Pimpl模式**: 核心类使用`AWAKENING_IMPL_DEFINITION`隐藏实现
3. **观察者模式**: 数据通过共享内存(SWMR)在任务间传递
4. **状态机模式**: AutoAimFsmController管理瞄准策略
5. **策略模式**: 根据FSM状态选择不同的瞄准算法

## 快速使用指南

### 1. 环境准备

#### 硬件要求
- **相机**: 海康威视工业相机（MVision系列）
- **串口**: 与下位机通信的串口设备（如 `/dev/ttyACM0`）
- **GPU**（可选）: NVIDIA显卡，用于TensorRT加速
- **系统**: Linux（推荐Ubuntu 20.04+）

#### 软件依赖
- **必需**:
  - OpenCV 4.x
  - Eigen3
  - yaml-cpp
  - Ceres Solver
  - Boost.Asio
  - spdlog
  - clang/clang++ (编译器)
  - Ninja (构建工具)
  
- **可选**:
  - CUDA 11.x+ (TensorRT后端)
  - TensorRT 10.x (深度学习推理加速)
  - OpenVINO (Intel CPU推理)
  - ROS2 (可视化与TF发布)

#### 海康相机SDK
```bash
# 安装海康相机SDK到 /opt/MVS
# 环境变量已配置在 env.bash 中
```

### 2. 构建项目

```bash
# 进入项目目录
cd /path/to/awakening

# 设置环境变量（可选，run.sh会自动执行）
source env.bash

# 增量构建（推荐）
./run.sh build

# 完全重建（会删除build目录）
./run.sh rebuild

# 构建并运行
./run.sh run auto_aim <config_path> [debug]
```

构建产物：
- 可执行文件: `bin/auto_aim`
- 库文件: `lib/libawakening_utils.so`, `lib/libawakening_auto_aim.so`

### 3. 配置文件

配置文件位于 `config/` 目录，主要配置项：

#### 基础配置
```yaml
enemy_color: red              # 敌方颜色 (red/blue)
bullet_speed: 23.0            # 弹速 (m/s)
max_infer_num: 5              # 最大并发推理数
```

#### 串口配置
```yaml
serial:
  enable: true                # 是否启用串口
  device_name: /dev/ttyACM0   # 串口设备路径
  baud_rate: 115200           # 波特率
  char_size: 8                # 数据位
  read_buf_size: 4096         # 读缓冲区大小
```

#### 相机配置
```yaml
camera:
  hik_camera:
    target_sn: DA1094115      # 相机序列号
    width: 1440               # 图像宽度
    height: 1080              # 图像高度
    acquisition_frame_rate: 250  # 帧率
    exposure_time: 1500       # 曝光时间 (μs)
    gain: 16.9                # 增益
    format: "bgr"             # 输出格式
  
  camera_info:                # 相机内参（标定获得）
    camera_matrix:            # 内参矩阵
      data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    distortion_coefficients:  # 畸变系数
      data: [k1, k2, p1, p2, k3]
```

#### 坐标变换配置
```yaml
tf:
  camera_in_gimbal:           # 相机相对云台的安装位姿
    t: [0.14, 0.05, 0.0]      # 平移向量 (m)
    R: [1.0, 0.0, 0.0,        # 旋转矩阵 (行优先)
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0]
  shoot_in_gimbal:            # 发射点相对云台的位姿
    t: [0.2, 0.0, 0.0]
    R: [1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0]
```

#### 模型配置
```yaml
armor_detector:
  armor_infer:
    model_type: tup           # 模型类型
    conf_threshold: 0.2       # 置信度阈值
  net_detector:
    backend: tensorrt         # 推理后端 (tensorrt/openvino)
    tensorrt:
      model_path: ${ROOT_DIR}/model/opt-1208-001.onnx
      use_cuda_preproces: true
```

#### 跟踪器配置
```yaml
armor_tracker:
  lost_time_thres: 0.5        # 丢失时间阈值 (s)
  tracking_thres: 10          # 确认跟踪所需帧数
  match_gate: 1000.0          # 匹配门限
  esekf_iter_num: 5           # ESEKF迭代次数
```

### 4. 运行模式

#### 模式一：实时运行（相机+串口）
```bash
# 使用默认配置
./run.sh run auto_aim config/test.yaml

# 启用调试模式
./run.sh run auto_aim config/test.yaml true
```

**前提条件**:
- 相机已连接且序列号匹配
- 串口设备可访问（可能需要sudo权限）
- 模型文件已放置在 `model/` 目录

#### 模式二：回放模式（离线数据）
```bash
# 修改配置文件启用player
# config/test.yaml:
# player:
#   enable: true
#   path: /path/to/record.bin

./run.sh run auto_aim config/test.yaml
```

#### 模式三：录制模式
```bash
# 修改配置文件启用recorder
# config/test.yaml:
# recorder:
#   enable: true

./run.sh run auto_aim config/test.yaml
```
录制文件保存在 `record/auto_aim/` 目录，文件名格式：`YYYY-MM-DD_HH-MM-SS.bin`

### 5. Web调试界面

#### 启动Web服务器
```bash
# 在项目根目录运行
python web.py
```

#### 访问界面
打开浏览器访问：`http://localhost:8000`

#### 功能特性
- **实时视频流**: MJPEG格式，60 FPS
- **数据监控**: 跟踪状态、目标位置、FSM状态
- **日志查看**: 运行时日志实时显示
- **弹道可视化**: 子弹轨迹预测

#### 共享内存数据
Web服务器从以下共享内存读取数据：
- `/dev/shm/awakening_frame`: 图像数据（MJPEG格式）
- `/dev/shm/awakening_data.json`: 运行时数据
- `/dev/shm/awakening_log.json`: 日志数据

### 6. 调试技巧

#### 查看实时统计信息
程序每秒输出统计信息：
```
detect: 120 track: 118 found: 115 solve: 1000 serial: 200 camera: 120 avg_latency: 15.2 ms
```
- `detect`: 检测次数
- `track`: 跟踪次数
- `found`: 成功跟踪次数
- `solve`: 解算次数（1000Hz）
- `serial`: 串口接收次数
- `camera`: 相机帧数
- `avg_latency`: 平均延迟（ms）

#### 调整自动曝光
```yaml
auto_exposure:
  enable: true
  target_brightness: 25.0     # 目标亮度
  tolerance: 3.0              # 容差
  step_gain: 15.0             # 调节步进
  exposure_min: 100.0         # 最小曝光
  exposure_max: 3500.0        # 最大曝光
```

#### 调整FSM状态切换
```yaml
auto_aim_fsm:
  single_whole_up: 1.5        # 单板→整车 阈值 (rad/s)
  whole_pair_up: 6.5          # 整车→双板 阈值
  pair_center_up: 16.5        # 双板→中心 阈值
  transfer_thresh: 50         # 状态切换确认帧数
```

### 7. 常见问题

#### Q1: 相机无法打开
```bash
# 检查相机连接
ls /dev/video*

# 检查相机序列号
# 在 config/*.yaml 中修改 target_sn
```

#### Q2: 串口权限不足
```bash
# 添加用户到dialout组
sudo usermod -a -G dialout $USER
# 注销重新登录生效
```

#### Q3: TensorRT模型加载失败
```bash
# 检查CUDA版本
nvcc --version

# 检查TensorRT版本
cat /usr/include/NvInferVersion.h

# 确保模型路径正确（${ROOT_DIR}会自动替换）
```

#### Q4: 编译错误
```bash
# 确保使用clang编译
export CC=clang
export CXX=clang++

# 清理重新构建
./run.sh rebuild
```

#### Q5: 运行时延迟过高
- 减少 `max_infer_num` 以降低GPU负载
- 降低相机分辨率或帧率
- 检查CPU/GPU占用率
- 调整跟踪器参数 `esekf_iter_num`

### 8. 性能优化建议

1. **推理加速**:
   - 使用TensorRT后端（GPU加速）
   - 启用CUDA预处理: `use_cuda_preproces: true`
   - 调整 `max_infer_num` 平衡并发与延迟

2. **相机优化**:
   - 使用硬件触发同步（多相机场景）
   - 调整曝光时间避免过曝/欠曝
   - ROI裁剪减少数据传输

3. **跟踪优化**:
   - 减少 `esekf_iter_num` 降低计算量
   - 调整 `match_gate` 提高匹配效率

4. **系统优化**:
   - 设置CPU性能模式: `cpufreq-set -g performance`
   - 绑定CPU核心: `taskset -c 0-3 ./bin/auto_aim`
   - 提高进程优先级: `nice -n -10 ./bin/auto_aim`

### 9. 开发与扩展

#### 添加新的任务节点
```cpp
// 参考 src/runtime/auto_aim.cpp
s.register_task<InputIO, OutputIO>("task_name", [&](InputIO::second_type&& input) {
    // 处理逻辑
    return std::make_tuple(std::optional<OutputIO::second_type>(output));
});
```

#### 添加新的坐标系
```cpp
// 在 SimpleFrame 枚举中添加
enum class SimpleFrame : int { 
    ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, 
    NEW_FRAME,  // 新坐标系
    N 
};

// 添加坐标变换边
tf->add_edge(SimpleFrame::GIMBAL, SimpleFrame::NEW_FRAME);
tf->push(SimpleFrame::GIMBAL, SimpleFrame::NEW_FRAME, timestamp, transform);
```

#### 自定义调试信息
```cpp
// 修改 src/tasks/auto_aim/debug.cpp
void draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx) {
    // 添加自定义绘制逻辑
}
```

### 10. 参考资料

- **项目文档**: `AGENTS.md`
- **配置示例**: `config/test.yaml`, `config/omni.yaml`
- **模型文件**: `model/` 目录
- **ROS2集成**: `src/_rcl/` 目录
- **Web界面**: `web.py`, `templates/`, `static/`

---

**注意**: 使用前请确保已正确配置相机内参、坐标变换等参数，否则会影响跟踪精度。建议先在调试模式下验证系统运行正常。

## wust_vl 通用算法库使用指南

### 一、库概述

`wust_vl` 是一个模块化的视觉算法库，提供相机驱动、神经网络推理、控制算法和通用工具。

### 二、模块结构

```
wust_vl/
├── include/
│   ├── video/          # 相机驱动（HIK/UVC/视频播放）
│   ├── ml_net/         # 神经网络推理（TRT/OpenVINO/NCNN/ORT）
│   ├── algorithm/      # 算法（PnP/PID/SMC控制器）
│   └── common/         # 通用工具（并发/串口/日志/参数）
```

### 三、构建安装

```bash
cd wust_vl
source env.bash           # 加载环境变量
./run.sh                  # 编译安装到 /usr/local
./run.sh -r               # 清理重新编译
./run.sh -i               # 重新安装
./run.sh -d               # 完全卸载
```

### 四、集成到项目

#### 1. CMakeLists.txt 配置

```cmake
find_package(wust_vl REQUIRED)
target_link_libraries(your_target wust_vl::wust_vl)
```

#### 2. 头文件引用

```cpp
#include <wust_vl.hpp>  // 引入所有模块
// 或按需引入：
#include <video/video.hpp>
#include <ml_net/ml_net.hpp>
#include <algorithm/algorithm.hpp>
#include <common/common.hpp>
```

### 五、核心模块使用

#### 1. 相机模块 (video)

```cpp
#include <video/camera.hpp>

wust_vl::video::Camera camera;
YAML::Node config = YAML::LoadFile("camera.yaml");
camera.init(config);             // 初始化相机
camera.start();                  // 启动采集

// 方式1：主动读取
ImageFrame frame = camera.readImage();

// 方式2：回调模式
camera.setFrameCallback([](const ImageFrame& frame) {
    // 处理帧
});
```

**配置示例 (camera.yaml):**
```yaml
type: HIK_CAMERA  # 或 VIDEO_PLAYER / UVC
hik_camera:
  target_sn: "DA1094115"
  width: 1440
  height: 1080
  ExposureTime: 1500
  Gain: 16.9
  AcquisitionFrameRate: 250
  PixelFormat: "bgr"
```

#### 2. 神经网络推理 (ml_net)

**TensorRT 后端:**
```cpp
#include <ml_net/tensorrt/tensorrt_net.hpp>

wust_vl::ml_net::TensorRTNet net;
wust_vl::ml_net::TensorRTNet::Params params;
params.model_path = "model.onnx";
params.input_dims = {1, 3, 640, 640};

net.init(params);

// 推理
net.input2Device(input_data);
net.infer(input_data, net.getAContext());
float* output = net.output2Host();
```

**OpenVINO 后端:**
```cpp
#include <ml_net/openvino/openvino_net.hpp>

wust_vl::ml_net::OpenvinoNet net;
wust_vl::ml_net::OpenvinoNet::Params params;
params.model_path = "model.xml";
params.device_name = "CPU";  // 或 GPU
params.mode = ov::hint::PerformanceMode::LATENCY;

net.init(params, [](ov::preprocess::PrePostProcessor& ppp) {
    // 预处理配置
});

ov::Tensor output = net.infer(input_tensor);
```

#### 3. 串口通信 (common::drivers)

```cpp
#include <common/drivers/serial_driver.hpp>

wust_vl::common::drivers::SerialDriver serial;

wust_vl::common::drivers::SerialDriver::SerialPortConfig cfg;
cfg.baud_rate = 115200;
cfg.char_size = 8;

serial.init_port("/dev/ttyACM0", cfg);

serial.set_receive_callback([](const uint8_t* data, size_t len) {
    // 处理接收数据
});

serial.start();

// 发送数据
auto packet = wust_vl::common::drivers::toVector(your_struct);
serial.write(packet);
```

#### 4. 参数配置系统 (common::utils)

```cpp
#include <common/utils/parameter.hpp>

// 定义参数组
class MyConfig : public wust_vl::common::utils::SimpleConfigBase<MyConfig> {
public:
    static constexpr const char* kKey = "my_config";
    
    double threshold;
    int max_count;
    
    void load(const YAML::Node& node) override {
        loadOnceOrUpdate(node, threshold, 
            [](const YAML::Node& n, double& v) { v = n["threshold"].as<double>(); });
        loadOnceOrUpdate(node, max_count,
            [](const YAML::Node& n, int& v) { v = n["max_count"].as<int>(); });
    }
};

// 使用
auto param = wust_vl::common::utils::Parameter::create();
MyConfig config;
param->registerGroup(config);
param->loadFromFile("config.yaml");
```

#### 5. 日志系统 (common::utils)

```cpp
#include <common/utils/logger.hpp>

// 初始化日志
wust_vl::initLogger("INFO", "./logs", true, true, false);

// 使用日志
WUST_INFO("module_name") << "信息日志";
WUST_WARN("module_name") << "警告日志";
WUST_ERROR("module_name") << "错误日志";
WUST_DEBUG("module_name") << "调试日志";
WUST_THROW_ERROR("module_name") << "抛出异常";  // 抛出 runtime_error
```

#### 6. PnP 求解器 (algorithm)

```cpp
#include <algorithm/pnp_solver.hpp>

wust_vl::algorithm::PnPSolver solver(cv::SOLVEPNP_IPPE);

// 设置物体坐标系点
std::vector<cv::Point3f> object_points = {
    {-0.0675, 0.0275, 0},
    {0.0675, 0.0275, 0},
    {0.0675, -0.0275, 0},
    {-0.0675, -0.0275, 0}
};
solver.setObjectPoints("armor", object_points);

// 求解位姿
cv::Mat rvec, tvec;
bool success = solver.solvePnP(
    image_points, rvec, tvec, 
    "armor", camera_matrix, dist_coeffs
);
```

#### 7. PID 控制器 (algorithm::control)

```cpp
#include <algorithm/control/pid.hpp>

wust_vl::algorithm::control::PID<double> pid;
pid.setGains(1.0, 0.1, 0.05);
pid.setOutputLimits(-10.0, 10.0);
pid.setIntegratorLimit(5.0);
pid.setFixedDt(0.001);  // 1ms

double output = pid.update(setpoint, measurement);
```

#### 8. 弹道补偿器 (common::utils)

```cpp
#include <common/utils/trajectory_compensator.hpp>

auto compensator = wust_vl::common::utils::CompensatorFactory::createCompensator("resistance");

YAML::Node cfg;
cfg["gravity"] = 9.8;
cfg["resistance"] = 0.01;
cfg["iteration_times"] = 20;
compensator->load(cfg);

Eigen::Vector3d target_pos(3.0, 0.0, 1.5);
double pitch = 0.0;
compensator->compensate(target_pos, pitch, bullet_speed);

double fly_time = compensator->getFlyingTime(target_pos, bullet_speed);
```

#### 9. 线程池 (common::concurrency)

```cpp
#include <common/concurrency/ThreadPool.h>

wust_vl::common::concurrency::ThreadPool pool(4);  // 4线程

pool.enqueue([]() {
    // 任务逻辑
});

pool.waitUntilEmpty();  // 等待所有任务完成
```

#### 10. 时间队列 (common::concurrency)

```cpp
#include <common/concurrency/queues.hpp>

// 时间窗口队列（自动清理过期数据）
wust_vl::common::concurrency::TimedQueue<MyData> queue(1.0);  // 1秒有效期

queue.push(data);

MyData out;
if (queue.pop_valid(out)) {
    // 获取有效数据
}

// 有序队列（按帧ID排序，处理丢帧）
wust_vl::common::concurrency::OrderedQueue<Frame> ordered_queue(50, 200);
```

#### 11. 数据录制/回放 (common::utils)

```cpp
#include <common/utils/recorder.hpp>

// 定义写入器
class MyWriter : public wust_vl::common::utils::Writer<MyData> {
public:
    void write(std::ostream& os, const MyData& data) override {
        os.write(reinterpret_cast<const char*>(&data), sizeof(data));
    }
};

// 录制
wust_vl::common::utils::Recorder<MyData> recorder(
    "record.bin", 
    std::make_shared<MyWriter>()
);
recorder.start();
recorder.push(data);

// 回放
class MyParser : public wust_vl::common::utils::Parser<MyData> {
public:
    bool read(std::istream& is, MyData& out) override {
        return static_cast<bool>(is.read(reinterpret_cast<char*>(&out), sizeof(out)));
    }
};

wust_vl::common::utils::RecorderParser<MyData> player(
    "record.bin",
    std::make_shared<MyParser>()
);
player.open();

MyData data;
while (player.readNext(data)) {
    // 处理数据
}
```

### 六、编译选项

在 CMakeLists.txt 中控制编译哪些后端：

```cmake
option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_WITH_NCNN     "Enable NCNN backend"       ON)
option(BUILD_WITH_ORT      "Enable ORT backend"        ON)
```

### 七、典型应用场景

该库适合构建视觉系统，如自瞄系统、目标跟踪、工业检测等。它提供了：
- 统一的相机接口
- 多后端推理框架
- 线程安全的数据队列
- 弹道解算和控制算法
- 参数热加载和日志系统
