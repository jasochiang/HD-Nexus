import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
import matplotlib.pyplot as plt
import swanlab
from collections import Counter
from matplotlib.font_manager import FontProperties

EMOTION_LABEL = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking',
                         'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']

# 配置logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Initializing and pre-processing dataset...")

class CarDataset(Dataset):

    def __init__(self, csv_file, mode = None, state = None):

        self.path_df = pd.read_csv(csv_file)
        self.mode = mode

        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 224
        self.body_width = 224
        self.face_height = 224
        self.face_width = 224

        # 图像和关键点归一化参数的设置保持不变
        if state is not None:
            self.image_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
            self.image_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
            self.gesture_mean = torch.tensor(state['gesture_mean']).view(3, 1, 1)
            self.gesture_std = torch.tensor(state['gesture_std']).view(3, 1, 1)
            self.posture_mean = torch.tensor(state['posture_mean']).view(3, 1, 1)
            self.posture_std = torch.tensor(state['posture_std']).view(3, 1, 1)
        else:
            self.image_mean, self.image_std, self.gesture_mean, self.gesture_std, self.posture_mean, self.posture_std = [None] * 6

        # --- 核心重构：在初始化时预处理所有样本信息 ---
        self.samples = []
        for idx in range(len(self.path_df)):
            frames_path_rel, label_path_rel = self.path_df.iloc[idx]

            # 构建绝对路径
            base_dir = "/media/Data1/aide/"
            frames_path = os.path.join(base_dir, frames_path_rel)
            label_path = os.path.join(base_dir, label_path_rel)

            # --- 将耗时的文件操作移到此处 ---
            incar_path = os.path.join(frames_path, 'incarframes')
            front_path = os.path.join(frames_path, 'frontframes')
            left_path = os.path.join(frames_path, 'leftframes')
            right_path = os.path.join(frames_path, 'rightframes')

            # 检查路径是否存在，避免运行时错误
            if not os.path.isdir(incar_path):
                logger.warning(f"Directory not found, skipping sample: {incar_path}")
                continue

            # 获取并排序文件列表 (只执行一次！)
            try:
                incar_frames = sorted(
                    [os.path.join(incar_path, f) for f in os.listdir(incar_path) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.basename(x).split('.')[0]))
                front_frames = sorted(
                    [os.path.join(front_path, f) for f in os.listdir(front_path) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.basename(x).split('.')[0]))
                left_frames = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if f.endswith('.jpg')],
                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
                right_frames = sorted(
                    [os.path.join(right_path, f) for f in os.listdir(right_path) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.basename(x).split('.')[0]))
            except FileNotFoundError:
                logger.warning(f"One of the frame directories not found for {frames_path}, skipping.")
                continue

            # 将所有信息打包存入 self.samples 列表
            self.samples.append({
                "label_path": label_path,
                "frame_paths": {
                    "incar": incar_frames,
                    "front": front_frames,
                    "left": left_frames,
                    "right": right_frames,
                }
            })
        #logger.info(f"Dataset pre-processing complete. Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # --- `__getitem__` 现在变得极其轻量和快速 ---
        #logger.info(f"--- 开始加载样本，索引(index): {idx} ---")
        # 1. 直接从预处理好的列表中获取样本信息
        sample_info = self.samples[idx]

        label_path = sample_info["label_path"]
        frame_paths = sample_info["frame_paths"]

        # 读取JSON标签和姿态列表
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        pose_list = label_json['pose_list']

        # 2. 调用现在变得更快的 load_frames
        buffer, buffer_front, buffer_left, buffer_right, buffer_body, buffer_face, keypoints = self.load_frames(
            frame_paths, pose_list)

        # 修改：不再拼接环境多视角，而是分别处理每个视角
        if self.mode == 'train':
            # 对每个视角单独进行随机翻转
            buffer = self.randomflip(buffer)
            buffer_front = self.randomflip(buffer_front)
            buffer_left = self.randomflip(buffer_left)
            buffer_right = self.randomflip(buffer_right)
        
        # 转换为张量格式
        buffer = self.to_tensor(buffer)
        buffer_front = self.to_tensor(buffer_front)
        buffer_left = self.to_tensor(buffer_left)
        buffer_right = self.to_tensor(buffer_right)
        buffer_body = self.to_tensor(buffer_body)
        buffer_face = self.to_tensor(buffer_face)
        
        keypoints = keypoints.permute(2, 0, 1).contiguous()
        posture = keypoints[:, :, :26, ]
        gesture = keypoints[:, :, 94:, ]

        # 图片和关键点正则化
        if self.image_mean is not None:
            buffer = (buffer - self.image_mean) / self.image_std
            buffer_front = (buffer_front - self.image_mean) / self.image_std
            buffer_left = (buffer_left - self.image_mean) / self.image_std
            buffer_right = (buffer_right - self.image_mean) / self.image_std
            buffer_body = (buffer_body - self.image_mean) / self.image_std
            buffer_face = (buffer_face - self.image_mean) / self.image_std
        if self.posture_mean is not None:
            posture = (posture - self.posture_mean) / self.posture_std
            gesture = (gesture - self.gesture_mean) / self.gesture_std

        posture = torch.stack([posture],dim=-1)
        gesture = torch.stack([gesture], dim=-1)

        emotion_label = EMOTION_LABEL.index(label_json['emotion_label'].capitalize())
        behavior_label = DRIVER_BEHAVIOR_LABEL.index(label_json['driver_behavior_label'])
        context_label = SCENE_CENTRIC_CONTEXT_LABEL.index(label_json['scene_centric_context_label'])
        vehicle_label = VEHICLE_BASED_CONTEXT_LABEL.index(label_json['vehicle_based_context_label'])

        # 返回环境多视角作为字典，而不是拼接后的张量
        context_views = {
            "incar": buffer,
            "front": buffer_front,
            "left": buffer_left,
            "right": buffer_right
        }

        return posture, gesture, context_views, buffer_body, buffer_face, emotion_label, behavior_label, context_label, vehicle_label

    def load_frames(self, frame_paths, pose_list):
        # --- 这个函数现在只负责读取和处理，不再进行文件查找和排序 ---

        frames = frame_paths['incar']
        front_frames = frame_paths['front']
        left_frames = frame_paths['left']
        right_frames = frame_paths['right']

        buffer, buffer_front, buffer_left, buffer_right = [], [], [], []
        buffer_body, buffer_face, keypoints = [], [], []

        # --- 优化后的帧采样逻辑 ---
        # 直接使用 range 的步长功能，更清晰高效
        for i in range(0, min(len(frames), 45)):
            if not i == 0 and not i % 3 == 2:
                continue
            img = cv2.imread(frames[i])
            front_img = cv2.imread(front_frames[i])
            left_img = cv2.imread(left_frames[i])
            right_img = cv2.imread(right_frames[i])

            # 如果任何一张图片读取失败，则跳过此帧
            if img is None or front_img is None or left_img is None or right_img is None:
                continue

            body = pose_list[i]['result'][0]['bbox']
            face = pose_list[i]['result'][0]['face_bbox']
            keypoint = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)

            img_body = img[int(body[1]):int(body[1] + max(body[3], 20)), int(body[0]):int(body[0] + max(body[2], 10))]
            img_face = img[int(face[1]):int(face[1] + max(face[3], 10)), int(face[0]):int(face[0] + max(face[2], 10))]
            i = 0
            # 改进的img_face异常处理
            if img_face.size == 0:  # 检查数组是否为空，而不是等待resize抛出异常
                # 如果脸部区域无效，创建一个黑色占位符图像
                img_face = np.zeros((self.face_height, self.face_width, 3), dtype=np.uint8)
                i = i+1
                print(f"无效次数{i}")
            else:
                img_face = cv2.resize(img_face, (self.face_width, self.face_height))

            img = cv2.resize(img, (self.resize_width, self.resize_height))
            front_img = cv2.resize(front_img, (self.resize_width, self.resize_height))
            left_img = cv2.resize(left_img, (self.resize_width, self.resize_height))
            right_img = cv2.resize(right_img, (self.resize_width, self.resize_height))
            img_body = cv2.resize(img_body, (self.body_width, self.body_height))

            buffer.append(torch.from_numpy(img).float())
            buffer_front.append(torch.from_numpy(front_img).float())
            buffer_left.append(torch.from_numpy(left_img).float())
            buffer_right.append(torch.from_numpy(right_img).float())
            buffer_body.append(torch.from_numpy(img_body).float())
            buffer_face.append(torch.from_numpy(img_face).float())
            keypoints.append(torch.from_numpy(keypoint).float())

        # 如果循环后buffer为空（例如所有帧都读取失败），需要返回正确形状的空张量以防止崩溃
        if not buffer:
            # 返回一个零张量或根据需要处理，这里简单地重复第一帧来避免错误
            # 在实际应用中，您可能需要更复杂的逻辑来处理损坏的数据
            raise RuntimeError(f"Could not load any valid frames for a sample. Check data integrity.")

        return torch.stack(buffer), torch.stack(buffer_front), torch.stack(buffer_left), \
            torch.stack(buffer_right), torch.stack(buffer_body), torch.stack(buffer_face), torch.stack(keypoints)

    def randomflip(self, buffer):

        if np.random.random() < 0.5:
            # flipped_buffer 的形状仍然不变，但图片在高度方向上被翻转了（NHWC）第0维是数量，第1维是高度，第2维是宽度，第3维是通道数
            buffer = torch.flip(buffer, dims=[1])

        if np.random.random() < 0.5:
            buffer = torch.flip(buffer, dims=[2])

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        # 卷积层处理的输入数据通常是 (C, N, H, W)，contiguous()方法可以使得数据在内存中连续存储，permute()方法可以交换维度的顺序
        return buffer.permute(3, 0, 1, 2).contiguous()


class CarDataCollator:
    def __init__(self):
        """
        数据整理器，负责将批次数据整理为模型可用的格式
        """
        pass

    def __call__(self, batch):
        posture, gesture, context_views, body, face, emotion_label, behavior_label, context_label, vehicle_label = zip(*batch)

        # 堆叠和填充批次数据
        posture = pad_sequence(posture, batch_first=True)
        gesture = pad_sequence(gesture, batch_first=True)
        
        # 处理环境多视角字典
        context_dict = {
            "incar": torch.stack([item["incar"] for item in context_views]),
            "front": torch.stack([item["front"] for item in context_views]),
            "left": torch.stack([item["left"] for item in context_views]),
            "right": torch.stack([item["right"] for item in context_views])
        }
        
        body = torch.stack(body)
        face = torch.stack(face)

        # 转换标签为张量
        emotion_label = torch.tensor(emotion_label)
        behavior_label = torch.tensor(behavior_label)
        context_label = torch.tensor(context_label)
        vehicle_label = torch.tensor(vehicle_label)

        # 组装批次数据
        batch_dict = {
            "posture": posture,
            "gesture": gesture,
            "context_views": context_dict,
            "body": body,
            "face": face,
            "emotion_label": emotion_label,
            "behavior_label": behavior_label,
            "context_label": context_label,
            "vehicle_label": vehicle_label
        }
        """
        # 如果启用了Mixup，应用Clip-级Mixup
        if self.mixup_alpha > 0:
            from dataset_utils import apply_clip_mixup
            batch_dict = apply_clip_mixup(batch_dict, self.mixup_alpha)
        """
        return batch_dict

def count_labels(dataset):
    """
    Calculate the distribution of categories in the dataset
    
    Args:
        dataset: CarDataset instance
        
    Returns:
        dict: Dictionary containing category counts
    """
    logger.info("Starting to count category distribution...")
    
    emotion_counts = Counter()
    behavior_counts = Counter()
    context_counts = Counter()
    vehicle_counts = Counter()
    
    for idx in range(len(dataset)):
        sample_info = dataset.samples[idx]
        label_path = sample_info["label_path"]
        
        # 读取JSON标签
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        
        # 统计各类别
        emotion_label = label_json['emotion_label'].capitalize()
        behavior_label = label_json['driver_behavior_label']
        context_label = label_json['scene_centric_context_label']
        vehicle_label = label_json['vehicle_based_context_label']
        
        emotion_counts[emotion_label] += 1
        behavior_counts[behavior_label] += 1
        context_counts[context_label] += 1
        vehicle_counts[vehicle_label] += 1
    
    # 整理结果
    result = {
        'emotion': {label: emotion_counts.get(label, 0) for label in EMOTION_LABEL},
        'behavior': {label: behavior_counts.get(label, 0) for label in DRIVER_BEHAVIOR_LABEL},
        'context': {label: context_counts.get(label, 0) for label in SCENE_CENTRIC_CONTEXT_LABEL},
        'vehicle': {label: vehicle_counts.get(label, 0) for label in VEHICLE_BASED_CONTEXT_LABEL}
    }
    
    logger.info("Category distribution count completed")
    return result

def create_label_distribution_charts(label_counts, save_dir=None):
    """
    Create category distribution bar charts
    
    Args:
        label_counts: Dictionary containing category counts
        save_dir: Directory path to save charts
        
    Returns:
        dict: Dictionary containing chart objects
    """
    logger.info("Creating category distribution charts...")
    
    # Create charts
    charts = {}
    categories = ['emotion', 'behavior', 'context', 'vehicle']
    titles = {
        'emotion': 'Emotion Distribution',
        'behavior': 'Driver Behavior Distribution',
        'context': 'Scene Centric Context Distribution',
        'vehicle': 'Vehicle Based Context Distribution'
    }
    
    for category in categories:
        plt.figure(figsize=(10, 6))
        counts = label_counts[category]
        labels = list(counts.keys())
        values = list(counts.values())
        
        bars = plt.bar(labels, values, color='skyblue')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(value),
                ha='center',
                va='bottom'
            )
        
        plt.title(titles[category])
        plt.xlabel('Categories')
        plt.ylabel('Sample Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save charts
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{category}_distribution.png"))
        
        charts[category] = plt.gcf()
    
    logger.info("Charts creation completed")
    return charts

def upload_label_distribution_to_swanlab(label_counts, charts, experiment_name="AIDE_base"):
    """
    Upload category distribution data and charts to SwanLab
    
    Args:
        label_counts: Dictionary containing category counts
        charts: Dictionary containing chart objects
        experiment_name: SwanLab experiment name
    """
    logger.info(f"Uploading category distribution data to SwanLab, experiment name: {experiment_name}")
    
    # Initialize SwanLab
    swanlab.init(project="AIDE_base", name=experiment_name)
    
    # Upload category count data
    metrics = {}
    for category, counts in label_counts.items():
        for label, count in counts.items():
            metrics[f"class_distribution_test/{category}/{label}"] = count
    
    # Upload charts
    for category, chart in charts.items():
        metrics[f"distribution_charts_test/{category}"] = swanlab.Image(chart)
    
    # Log metrics
    swanlab.log(metrics)
    
    # Finish experiment
    swanlab.finish()
    logger.info("Data successfully uploaded to SwanLab")

def test_dataset_and_dataloader():

        with open("keypoints_stats.json", "r") as f:
            stats = json.load(f)

        train_dataset = CarDataset(csv_file='/media/Data1/zwj/training.csv', mode=None, state=stats)
        
        # Count label distribution

        label_counts = count_labels(train_dataset)
        print("Label distribution statistics:")
        for category, counts in label_counts.items():
            print(f"\n{category.capitalize()} category distribution:")
            for label, count in counts.items():
                print(f"  {label}: {count}")

        # Create bar charts
        #charts = create_label_distribution_charts(label_counts, save_dir="label_distribution_charts")
        
        # Upload to SwanLab
        #upload_label_distribution_to_swanlab(label_counts, charts)
        
        # Create collator instance
        collator = CarDataCollator()

        # 使用 collator 创建 DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=False, collate_fn=collator)

        # 测试 collator
        for batch_idx, batch in enumerate(train_dataloader):
            print("Posture shape:", batch["posture"].shape)
            print("Gesture shape:", batch["gesture"].shape)
            print("Context views shapes:")
            for view_name, view_tensor in batch["context_views"].items():
                print(f"  {view_name}: {view_tensor.shape}")
            print("Body shape:", batch["body"].shape)
            print("Face shape:", batch["face"].shape)
            print("Emotion label shape:", batch["emotion_label"].shape)
            print("Behavior label shape:", batch["behavior_label"].shape)
            print("Context label shape:", batch["context_label"].shape)
            print("Vehicle label shape:", batch["vehicle_label"].shape)

            #print(batch["posture"][0, 0, :5])  # 打印前5个关键点
            
            break

if __name__ == "__main__":
    # Get user input
    user_input = input("Enter 1 to run tests, or 2 to only count category distribution, or anything else to exit: ")

    if user_input == "1":
        print("Starting tests...")
        test_dataset_and_dataloader()
    elif user_input == "2":
        print("Starting category distribution count...")
        with open("keypoints_stats.json", "r") as f:
            stats = json.load(f)
        test_dataset = CarDataset(csv_file='/xxx/xxx', mode=None, state=stats)
        label_counts = count_labels(test_dataset)
        charts = create_label_distribution_charts(label_counts, save_dir="label_distribution_charts")
        upload_label_distribution_to_swanlab(label_counts, charts)
    else:
        print("Thank you")
        exit()
