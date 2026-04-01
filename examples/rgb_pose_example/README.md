# RGB Pose

## 概要
在做VLN模型无instruction训练时需要的数据可以抽象为一系列轨迹，一个轨迹包含一系列的RGB图像和对应的相机位姿。而互联网上存在的各种不同的数据源对数据的组织形式不一致，[`rgb_pose_to_lerobot.py`](../../rgb_pose_to_lerobot.py)可以将各个数据源的数据转换为本仓库使用的统一的格式。

## 抽象

[`rgb_pose_to_lerobot.py`](../../rgb_pose_to_lerobot.py)主要靠[`rgb_pose_dataset.py`](../../utils/rgb_pose_dataset.py)中的`RGBPoseTrajectorySource`来提供从具体某个数据源读取轨迹数据的能力。其中的`RGBPoseTrajectory`提供从具体的一条轨迹中返回每一个轨迹点数据的能力。

对于每一个数据源，只需要按照数据源的格式继承并实现这两个类即可，一个具体的例子见[`real_walking_source.py`](./real_walking_source.py)，该例子处理`Video2Poses`仓库生成的数据。


实际使用如下指令转化数据：
```bash
uv run rgb_pose_to_lerobot.py \
    --raw_dir path/to/Video2Poses-input-dir \
    --source_cls examples.rgb_pose_example.real_walking_source:RealWalkingRGBPoseSource \
    --output_dir path/to/output-dir \
    --dataset_name dataset_name \
    --num_processes 1
```

该命令会使用`examples.rgb_pose_example.real_walking_source:RealWalkingRGBPoseSource`这个`RGBPoseTrajectory`来从`raw_dir`这个数据源读取数据，并将数据存到`path/to/output-dir/dataset_name`中。

可以通过如下命令可视化生成的轨迹：
```bash
uv run examples/rgb_pose_example/visualize_episode.py --dataset_dir path/to/dataset --episode_index episode_index --output output_name.mp4
```

## Supported Formats

### 1. Video2Poses（`RealWalkingRGBPoseSource`）

适用于 [Video2Poses](https://github.com/XieWeikai/Video2Poses) 仓库生成的数据。每条轨迹对应一个 `*-camera.json`文件和一个 `.mp4` 视频文件。

**输入目录结构：**
```
raw_dir/
├── video_id_1-camera.json   # 含每帧的 4×4 cam2world 位姿、内参、坐标系约定
├── video_id_2-camera.json
└── ...
```

- **位姿格式**：4×4 齐次变换矩阵（cam2world），OpenCV 相机坐标系
- **FPS**：由 manifest 中的 `sample_fps` 字段指定
- **内参**：manifest 中每帧提供 `fx, fy, cx, cy`

**示例命令：**
```bash
uv run rgb_pose_to_lerobot.py \
    --raw_dir path/to/video2poses-output \
    --source_cls examples.rgb_pose_example.real_walking_source:RealWalkingRGBPoseSource \
    --output_dir ./tmp --dataset_name my-walking-dataset
```

### 2. CityWalker（`CityWalkerRGBPoseSource`）

适用于 [CityWalker](https://github.com/ai4ce/CityWalker) 数据集。轨迹由文本格式的位姿文件和逐帧 JPEG 图片组成，位姿来自 DPVO 单目视觉里程计。

**输入目录结构：**
```
raw_dir/
├── pose_label/
│   ├── pose_traj_01.txt      # 每 3 行一帧：GPS / 位姿 / 场景类别
│   ├── pose_traj_02.txt
│   └── ...
└── obs/
    ├── traj_nav_01/
    │   ├── forward_0000.jpg  # 400×400 RGB
    │   ├── forward_0001.jpg
    │   └── ...
    ├── traj_nav_02/
    └── ...
```

- **位姿格式**：6D `[tx, ty, tz, rx, ry, rz]`，平移单位为米，旋转为轴角（弧度），OpenCV 相机坐标系
- **FPS**：1（位姿标注采样率约每秒一帧）
- **内参**：无（DPVO 不依赖标定）

**示例命令：**
```bash
uv run rgb_pose_to_lerobot.py \
    --raw_dir path/to/CityWalker \
    --source_cls examples.rgb_pose_example.citywalker_source:CityWalkerRGBPoseSource \
    --output_dir path/to/output --dataset_name citywalker-lerobot
```
