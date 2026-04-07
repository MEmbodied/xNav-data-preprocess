# xNav Data Preprocess

## 环境配置
本仓库使用[`uv`](https://docs.astral.sh/uv/getting-started)管理环境，安装好uv后使用如下命令配置环境

```bash
uv sync --all-groups
```

该命令会将环境装在项目目录下的`.venv`中，之后使用`uv run xxx.py`即可用该环境跑某个python程序。

## 概述
本仓库包含xNav对一些开源数据集的格式转换，详见[`docs`](./docs/)。

- [`rgb_pose_to_lerobot.py`](rgb_pose_to_lerobot.py): 给第一视角web video封装了一份代码，详见[`rgb_pose_example/README.md`](examples/rgb_pose_example/README.md)。
- [`lerobot_creator_example.py`](lerobot_creator_example.py): 教程示例代码。


