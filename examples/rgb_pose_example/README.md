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


