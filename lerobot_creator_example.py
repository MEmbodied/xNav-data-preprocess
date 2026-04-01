"""极简教学示例：如何使用 LeRobotCreator + Trajectories + Traj 创建数据集。

本示例使用 MockTrajectories 和 MockTraj 生成合成数据，演示以下内容：
1. 如何定义 FEATURES schema（含三个视角的 video）
2. 如何实现 Trajectories 和 Traj 接口
3. 如何配置并调用 LeRobotCreator 完成数据集导出

本示例包含三个相机视角：
- video.ego_view      — 正前方视野
- video.left_front    — 左前方视野
- video.right_front   — 右前方视野

每个视角都有独立的内参(K)和外参(T_body<-camera)。

运行方式：
    uv run lerobot_creator_example.py --output_dir ./tmp --dataset_name mock-demo
"""

import logging
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from utils import Trajectories, Traj
from utils.lerobot.lerobot_creater import LeRobotCreator

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Schema 定义
# ---------------------------------------------------------------------------
# POSE_AXES 定义了 observation.state 和 action 的语义：
#   前三维 (tx, ty, tz) 为平移，单位 m
#   后四维 (qx, qy, qz, qw) 为四元数表示的旋转
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

# 合成数据参数
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
NUM_EPISODES = 3
FRAMES_PER_EPISODE = 20
FPS = 10

# ---------------------------------------------------------------------------
# features schema
# ---------------------------------------------------------------------------
# features 字典完整描述了数据集中每一帧包含哪些字段，以及每个字段的类型和形状。
# LeRobotCreator 会据此：
#   - 生成 meta/info.json 中的 schema
#   - 决定哪些字段走 video 编码（dtype="video"），哪些存入 parquet
#
# 注意：timestamp、frame_index、episode_index、index、task_index
#       这几个字段由 LeRobotCreator 自动生成，不需要在 features 中声明，
#       也不需要在 Traj.__iter__ 的 frame dict 中提供。
FEATURES = {
    # 任务描述索引，由 creator.add_task(task_string) 获取
    "annotation.human.action.task_description": {
        "dtype": "int32",
        "shape": (1,),
        "names": None,
    },
    # 机体位姿：7D [tx, ty, tz, qx, qy, qz, qw]
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": POSE_AXES},
    },
    # 正前方视野（RGB video）
    "video.ego_view": {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    },
    # 左前方视野
    "video.left_front": {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    },
    # 右前方视野
    "video.right_front": {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    },
    # 动作，格式同 observation.state
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": POSE_AXES},
    },
}


# ---------------------------------------------------------------------------
# 相机参数（合成）
# ---------------------------------------------------------------------------
# 每个视角的内参矩阵 K（pinhole 模型）
CAMERA_INTRINSICS = {
    "video.ego_view": [500.0, 500.0, 320.0, 240.0],       # [fx, fy, cx, cy]
    "video.left_front": [480.0, 480.0, 320.0, 240.0],
    "video.right_front": [480.0, 480.0, 320.0, 240.0],
}

# 每个视角的外参 T_body<-camera（4×4 齐次变换矩阵）
# 含义：将相机坐标系下的点变换到机体坐标系
#   body frame: +X 前, +Y 左, +Z 上
#   camera frame (OpenCV): +X 右, +Y 下, +Z 前
#   目前设定机体坐标系和相机坐标系的原点重合，实际应用中通常会有安装偏移（T_body<-camera 中的平移部分）
BODY_FROM_CAMERA = {
    # 正前方相机：OpenCV -> body 的标准变换
    "video.ego_view": np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32),
    # 左前方相机：相对 ego_view 绕 body Z 轴旋转 +45°，安装偏移 (0, 0.3, 0)
    "video.left_front": np.array([
        [0, 0, 1, 0],
        [-0.707, 0.707, 0, 0.3],
        [-0.707, -0.707, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32),
    # 右前方相机：相对 ego_view 绕 body Z 轴旋转 -45°，安装偏移 (0, -0.3, 0)
    "video.right_front": np.array([
        [0, 0, 1, 0],
        [-0.707, -0.707, 0, -0.3],
        [0.707, -0.707, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# MockTraj — 单条轨迹
# ---------------------------------------------------------------------------
class MockTraj(Traj):
    """一条合成轨迹。

    Traj 的核心职责：
    1. __len__  返回帧数
    2. __iter__ 逐帧 yield (frame_dict, task_string)
       frame_dict 的 key 必须与 FEATURES 中的非自动字段一一对应
    3. metadata  返回 per-episode 的元信息，会被写入 episodes_extras.jsonl
    """

    def __init__(self, episode_index: int, task: str, task_idx: int):
        self._episode_index = episode_index
        self._task = task
        self._task_idx = task_idx
        self._num_frames = FRAMES_PER_EPISODE

    def __len__(self) -> int:
        return self._num_frames

    @property
    def metadata(self) -> dict:
        """Per-episode 元信息。

        推荐在此记录每个视角的内参和外参，方便下游使用时恢复相机模型。
        格式约定：
            "{video_key}.K"                  — 该视角的内参 [fx, fy, cx, cy]
            "{video_key}.body_from_camera"   — T_{body<-camera}
        """
        meta = {}
        for video_key in ["video.ego_view", "video.left_front", "video.right_front"]:
            meta[f"{video_key}.K"] = CAMERA_INTRINSICS[video_key]
            meta[f"{video_key}.body_from_camera"] = BODY_FROM_CAMERA[video_key]
        return meta

    def __iter__(self) -> Iterable[Tuple[dict, str]]:
        """逐帧产出数据。

        每次 yield 一个 tuple: (frame_dict, task_string)

        frame_dict 中：
        - 数值字段（observation.state, action 等）直接给 numpy array
        - 视频字段（video.*）给 uint8 numpy array，shape = (H, W, 3)
        - annotation.human.action.task_description 给 int32 array，值为 task_idx
        """
        for i in range(self._num_frames):
            # 合成位姿：沿 X 轴匀速前进，无旋转（单位四元数 [0,0,0,1]）
            t = i / FPS
            state = np.array(
                [t * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )

            # 合成图像：为每个视角生成带颜色区分的测试图
            ego_img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), [100, 150, 200], dtype=np.uint8)
            left_img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), [200, 100, 150], dtype=np.uint8)
            right_img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), [150, 200, 100], dtype=np.uint8)

            frame = {
                "annotation.human.action.task_description": np.array([self._task_idx], dtype=np.int32),
                "observation.state": state,
                "video.ego_view": ego_img,
                "video.left_front": left_img,
                "video.right_front": right_img,
                "action": state.copy(),
            }
            yield frame, self._task


# ---------------------------------------------------------------------------
# MockTrajectories — 轨迹集合
# ---------------------------------------------------------------------------
class MockTrajectories(Trajectories):
    """合成轨迹集合。

    Trajectories 的核心职责：
    1. 类变量 FPS / ROBOT_TYPE / FEATURES 定义数据集全局属性
    2. __init__  接收数据路径（本示例忽略）和 get_task_idx 回调
    3. __len__   返回轨迹（episode）总数
    4. __iter__  逐条 yield Traj 对象
    5. schema    返回 FEATURES，供 LeRobotCreator 初始化

    get_task_idx 是一个回调函数，签名为 (task_string) -> int，
    由外部传入（通常绑定到 creator.add_task），用于将任务文本
    注册到 LeRobot 的 tasks.jsonl 中并获取对应的整数索引。
    """

    FPS = FPS
    ROBOT_TYPE = "lerobot"
    FEATURES = FEATURES
    INSTRUCTION_KEY = "annotation.human.action.task_description"

    def __init__(self, data_path: str, get_task_idx=None):
        """
        Args:
            data_path: 原始数据路径（本示例中不使用）
            get_task_idx: 回调函数，将 task 文本映射为整数索引
        """
        self._get_task_idx = get_task_idx or (lambda t: 0)
        self._num_episodes = NUM_EPISODES

    def __len__(self) -> int:
        return self._num_episodes

    def __iter__(self) -> Iterable[MockTraj]:
        for i in range(self._num_episodes):
            task = "walk forward"
            task_idx = self._get_task_idx(task)
            yield MockTraj(episode_index=i, task=task, task_idx=task_idx)

    @property
    def schema(self) -> dict:
        return self.FEATURES


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="LeRobotCreator 教学示例")
    parser.add_argument("--output_dir", type=str, default="./tmp", help="输出目录")
    parser.add_argument("--dataset_name", type=str, default="mock-demo", help="数据集名称")
    parser.add_argument("--num_processes", type=int, default=2, help="工作进程数")
    parser.add_argument("--codec", type=str, default="h264", help="视频编码器")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / args.dataset_name

    # -----------------------------------------------------------------------
    # Step 1: 创建 LeRobotCreator
    # -----------------------------------------------------------------------
    # LeRobotCreator 是多进程数据写入引擎，负责：
    #   - 将帧数据写入 parquet 文件
    #   - 将图像帧编码为 mp4 视频
    #   - 维护 meta 目录下的所有元数据文件
    creator = LeRobotCreator(
        root=str(root),
        robot_type=MockTrajectories.ROBOT_TYPE,
        fps=MockTrajectories.FPS,
        features=MockTrajectories.FEATURES,
        num_workers=max(1, args.num_processes),
        num_video_encoders=max(1, int(args.num_processes * 1.75)),
        codec=args.codec,
        pix_fmt="yuv420p",
        has_extras=True,  # 启用 per-episode metadata（写入 episodes_extras.jsonl）
    )

    # -----------------------------------------------------------------------
    # Step 2: 实例化 Trajectories
    # -----------------------------------------------------------------------
    # get_task_idx 回调绑定到 creator.add_task，
    # 这样 Traj 在构造时就能拿到正确的 task index
    trajectories = MockTrajectories(
        data_path="unused",
        get_task_idx=creator.add_task,
    )

    # -----------------------------------------------------------------------
    # Step 3: 逐条提交轨迹
    # -----------------------------------------------------------------------
    # submit_episode 接收一个可迭代对象（即 Traj），
    # 内部 worker 会调用其 __iter__ 逐帧消费数据。
    # 如果 Traj 有 .metadata 属性，会自动写入 episodes_extras.jsonl。
    start = time.time()
    for i, traj in enumerate(trajectories):
        creator.submit_episode(traj)
        logging.info("Submitted episode %d / %d", i + 1, len(trajectories))

    # -----------------------------------------------------------------------
    # Step 4: 等待所有 worker 和 encoder 完成
    # -----------------------------------------------------------------------
    # wait() 会阻塞直到：
    #   1. 所有 episode 处理完毕（parquet 写入完成）
    #   2. 所有视频编码完成（mp4 生成完毕）
    #   3. 所有元数据刷盘（info.json, tasks.jsonl, episodes.jsonl 等）
    creator.wait()

    elapsed = time.time() - start
    logging.info("Done! %d episodes in %.2fs -> %s", len(trajectories), elapsed, root)


if __name__ == "__main__":
    main()
