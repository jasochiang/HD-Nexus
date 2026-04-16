import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import swanlab
from typing import Dict, List, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F

def get_posture_adjacency_matrix():
    # 身体姿态 26 个关键点的邻接矩阵
    num_posture_keypoints = 26
    posture_adjacency_matrix = np.zeros((num_posture_keypoints, num_posture_keypoints), dtype=int)

    # 定义身体关键点的连接关系
    posture_connections = [
        (0, 1), (0, 2),  # Nose to LEye and REye
        (1, 3), (2, 4),  # LEye to LEar and REye to REar
        (18, 5), (18, 6),  # Neck to LShoulder and RShoulder
        (5, 7), (7, 9),  # LShoulder to LElbow, LElbow to LWrist
        (6, 8), (8, 10),  # RShoulder to RElbow, RElbow to RWrist
        (11, 13), (13, 15),  # LHip to LKnee, LKnee to LAnkle
        (12, 14), (14, 16),  # RHip to Rknee, Rknee to RAnkle
        (18, 19), # Neck to Hip
        (13, 24), (14, 25),  # LKnee to LAnkle, RKnee to RAnkle
        (11, 19), (12, 19),  # LHip and RHip to Hip
        (15, 24), (15, 20), (15, 22),  # LAnkle to LHeel, LBigToe, LSmallToe
        (16, 25), (16, 21), (16, 23),  # RAnkle to RHeel, RBigToe, RSmallToe
        (18, 17), (17, 0)  # Neck to Head, Head to Nose
    ]

    # 在邻接矩阵中标记连接
    for i, j in posture_connections:
        posture_adjacency_matrix[i, j] = 1
        posture_adjacency_matrix[j, i] = 1  # 对称连接

    #print("Body Adjacency Matrix:\n", posture_adjacency_matrix)
    #print("维度:\n", posture_adjacency_matrix.shape)

    return posture_adjacency_matrix

def get_gesture_adjacency_matrix():
    # 手势关键点的邻接矩阵（42 个关键点）
    num_gesture_keypoints = 42
    gesture_adjacency_matrix = np.zeros((num_gesture_keypoints, num_gesture_keypoints), dtype=int)

    # 左手和右手关键点的索引范围
    # left_hand_indices = range(94, 115)
    # right_hand_indices = range(115, 136)

    # 将左手和右手的索引映射到局部手势矩阵的索引 [0, 41]
    left_hand_local_indices = range(0, 21)  # 对应于左手的 21 个关键点
    right_hand_local_indices = range(21, 42)  # 对应于右手的 21 个关键点

    # 左手关键点的连接
    for i in range(20):
        gesture_adjacency_matrix[left_hand_local_indices[i], left_hand_local_indices[i + 1]] = 1
        gesture_adjacency_matrix[left_hand_local_indices[i + 1], left_hand_local_indices[i]] = 1

    # 右手关键点的连接
    for i in range(20):
        gesture_adjacency_matrix[right_hand_local_indices[i], right_hand_local_indices[i + 1]] = 1
        gesture_adjacency_matrix[right_hand_local_indices[i + 1], right_hand_local_indices[i]] = 1

    # 在左手和右手的根节点（关键点 #94 和 #115）之间添加连接
    gesture_adjacency_matrix[0, 21] = 1
    gesture_adjacency_matrix[21, 0] = 1

    #print("Gesture Adjacency Matrix:\n", gesture_adjacency_matrix)

    return gesture_adjacency_matrix

def normalize_adjacency_matrix(A):
    """
    对邻接矩阵 A 进行对称归一化
    :param A: 邻接矩阵，形状为 (N, N)
    :return: 归一化后的邻接矩阵
    """
    # 计算度矩阵 D
    D = np.sum(A, axis=1)  # 每个节点的度数 (N,)
    D_inv_sqrt = np.power(D, -0.5)  # 计算度数的 -1/2 次方
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0  # 处理度数为 0 的节点（孤立节点）

    # 创建度矩阵的逆平方根形式
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)  # 对角矩阵

    # 计算归一化的邻接矩阵,@ 符号用于 矩阵乘法
    A_normalized = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    A_normalized = torch.from_numpy(A_normalized).float()

    return A_normalized


def get_normalize_adjacency_matrix():

    # 对身体姿态的邻接矩阵进行归一化
    posture_adjacency_matrix_normalized = normalize_adjacency_matrix(get_posture_adjacency_matrix())
    # print("Normalized Body Adjacency Matrix:\n", body_adjacency_matrix_normalized)

    # 对手势的邻接矩阵进行归一化
    gesture_adjacency_matrix_normalized = normalize_adjacency_matrix(get_gesture_adjacency_matrix())
    # print("Normalized Gesture Adjacency Matrix:\n", gesture_adjacency_matrix_normalized)

    return posture_adjacency_matrix_normalized, gesture_adjacency_matrix_normalized


# 新增: SoftMoE专家利用率分析工具
class ExpertUtilizationAnalyzer:
    """分析SoftMoE专家利用率的工具类"""
    
    def __init__(self):
        # 存储收集到的组合权重
        self.driver_combine_weights = []
        self.environment_combine_weights = []
        
        # 存储专家-类别关联数据
        self.driver_expert_class_data = {
            "emotion": [],   # 情感分类任务
            "behavior": []   # 行为分类任务
        }
        self.environment_expert_class_data = {
            "context": [],   # 上下文分类任务
            "vehicle": []    # 车辆分类任务
        }
        
        # 类别名称
        self.class_names = {
            "emotion": ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger'],
            "behavior": ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking', 'Body Movement'],
            "context": ['Traffic Jam', 'Waiting', 'Smooth Traffic'],
            "vehicle": ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']
        }
    
    def collect_combine_weights(self, model_output, domain="driver"):
        """
        收集模型前向传播中的组合权重
        
        Args:
            model_output: 包含组合权重的字典或对象
            domain: 领域名称 ("driver" 或 "environment")
        """
        if domain == "driver":
            self.driver_combine_weights.append(model_output["driver_combine_weights"].detach().cpu())
        else:
            self.environment_combine_weights.append(model_output["environment_combine_weights"].detach().cpu())
    
    def collect_expert_class_data(self, model_output, labels, domain="driver", task="emotion"):
        """
        收集专家-类别关联数据
        
        Args:
            model_output: 包含组合权重的字典或对象
            labels: 真实标签
            domain: 领域名称 ("driver" 或 "environment")
            task: 任务名称 ("emotion", "behavior", "context" 或 "vehicle")
        """
        if domain == "driver":
            if task == "emotion":
                self.driver_expert_class_data["emotion"].append((
                    model_output["driver_combine_weights"].detach().cpu(),
                    labels.detach().cpu()
                ))
            elif task == "behavior":
                self.driver_expert_class_data["behavior"].append((
                    model_output["driver_combine_weights"].detach().cpu(),
                    labels.detach().cpu()
                ))
        else:
            if task == "context":
                self.environment_expert_class_data["context"].append((
                    model_output["environment_combine_weights"].detach().cpu(),
                    labels.detach().cpu()
                ))
            elif task == "vehicle":
                self.environment_expert_class_data["vehicle"].append((
                    model_output["environment_combine_weights"].detach().cpu(),
                    labels.detach().cpu()
                ))
    
    def compute_expert_importance(self):
        """
        计算每个专家的平均重要性分数
        
        Returns:
            driver_importance: 驾驶员领域专家重要性
            environment_importance: 环境领域专家重要性
        """
        # 合并所有批次的组合权重
        if self.driver_combine_weights:
            driver_weights = torch.cat(self.driver_combine_weights, dim=0)
            # 计算每个专家的平均重要性 (batch, seq_len, num_experts, slots) -> (num_experts,)
            driver_importance = driver_weights.mean(dim=(0, 1, 3)).numpy()
        else:
            driver_importance = None
        
        if self.environment_combine_weights:
            environment_weights = torch.cat(self.environment_combine_weights, dim=0)
            environment_importance = environment_weights.mean(dim=(0, 1, 3)).numpy()
        else:
            environment_importance = None
        
        return driver_importance, environment_importance
    
    def compute_expert_class_association(self):
        """
        计算专家-类别关联性
        
        Returns:
            字典，包含每个任务的专家-类别关联矩阵
        """
        results = {}
        
        # 处理驾驶员领域任务
        for task in ["emotion", "behavior"]:
            if not self.driver_expert_class_data[task]:
                continue
                
            # 获取类别数量
            num_classes = len(self.class_names[task])
            
            # 合并所有批次的数据
            all_weights = []
            all_labels = []
            for weights, labels in self.driver_expert_class_data[task]:
                all_weights.append(weights)
                all_labels.append(labels)
            
            weights = torch.cat(all_weights, dim=0)  # (total_batch, seq_len, num_experts, slots)
            labels = torch.cat(all_labels, dim=0)    # (total_batch,)
            
            # 获取专家数量
            num_experts = weights.shape[2]
            
            # 初始化关联矩阵
            association = np.zeros((num_experts, num_classes))
            
            # 计算每个类别的专家激活
            for cls_idx in range(num_classes):
                # 找到属于当前类别的样本
                mask = (labels == cls_idx)
                if not mask.any():
                    continue
                    
                # 提取这些样本的组合权重
                cls_weights = weights[mask]
                
                # 计算平均激活 (对批次和序列长度维度取平均)
                # 对于驾驶员领域，我们关注第一个槽位（情感任务）或第二个槽位（行为任务）
                slot_idx = 0 if task == "emotion" else 1
                cls_activation = cls_weights[:, :, :, slot_idx].mean(dim=(0, 1)).numpy()
                
                # 存储到关联矩阵
                association[:, cls_idx] = cls_activation
            
            results[task] = association
        
        # 处理环境领域任务
        for task in ["context", "vehicle"]:
            if not self.environment_expert_class_data[task]:
                continue
                
            # 获取类别数量
            num_classes = len(self.class_names[task])
            
            # 合并所有批次的数据
            all_weights = []
            all_labels = []
            for weights, labels in self.environment_expert_class_data[task]:
                all_weights.append(weights)
                all_labels.append(labels)
            
            weights = torch.cat(all_weights, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            # 获取专家数量
            num_experts = weights.shape[2]
            
            # 初始化关联矩阵
            association = np.zeros((num_experts, num_classes))
            
            # 计算每个类别的专家激活
            for cls_idx in range(num_classes):
                # 找到属于当前类别的样本
                mask = (labels == cls_idx)
                if not mask.any():
                    continue
                    
                # 提取这些样本的组合权重
                cls_weights = weights[mask]
                
                # 计算平均激活
                # 对于环境领域，我们关注第一个槽位（上下文任务）或第二个槽位（车辆任务）
                slot_idx = 0 if task == "context" else 1
                cls_activation = cls_weights[:, :, :, slot_idx].mean(dim=(0, 1)).numpy()
                
                # 存储到关联矩阵
                association[:, cls_idx] = cls_activation
            
            results[task] = association
        
        return results
    
    def visualize_expert_importance(self, save_dir=None):
        """
        可视化专家重要性分数
        
        Args:
            save_dir: 保存图像的目录（可选）
            
        Returns:
            driver_fig: 驾驶员领域专家重要性图
            environment_fig: 环境领域专家重要性图
        """
        driver_importance, environment_importance = self.compute_expert_importance()
        
        figures = {}
        
        # 可视化驾驶员领域专家重要性
        if driver_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(range(len(driver_importance))), y=driver_importance, ax=ax)
            ax.set_title("Rdriver_moe")
            ax.set_xlabel("Expert Index")
            ax.set_ylabel("Average Importance Score")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加均匀分布的参考线
            uniform_value = 1.0 / len(driver_importance)
            ax.axhline(y=uniform_value, color='r', linestyle='--', label=f'Uniform Distribution ({uniform_value:.3f})')
            ax.legend()
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/driver_expert_importance.png", dpi=300)
            
            figures["driver"] = fig
        
        # 可视化环境领域专家重要性
        if environment_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(range(len(environment_importance))), y=environment_importance, ax=ax)
            ax.set_title("Renvironment_moe")
            ax.set_xlabel("Expert Index")
            ax.set_ylabel("Average Importance Score")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加均匀分布的参考线
            uniform_value = 1.0 / len(environment_importance)
            ax.axhline(y=uniform_value, color='r', linestyle='--', label=f'Uniform Distribution ({uniform_value:.3f})')
            ax.legend()
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/environment_expert_importance.png", dpi=300)
            
            figures["environment"] = fig
        
        return figures
    
    def visualize_expert_class_association(self, save_dir=None):
        """
        可视化专家-类别关联性
        
        Args:
            save_dir: 保存图像的目录（可选）
            
        Returns:
            字典，包含每个任务的专家-类别关联图
        """
        associations = self.compute_expert_class_association()
        figures = {}
        
        for task, association in associations.items():
            if association is None or association.size == 0:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 使用热力图可视化关联矩阵
            sns.heatmap(
                association, 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu",
                xticklabels=self.class_names[task],
                yticklabels=[f"Expert {i}" for i in range(association.shape[0])],
                ax=ax
            )
            
            ax.set_title(f"{task.capitalize()} task Expert-Class Association")
            ax.set_ylabel("expert")
            ax.set_xlabel("class")
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/{task}_expert_class_association.png", dpi=300)
            
            figures[task] = fig
        
        return figures
    
    def get_swanlab_metrics(self):
        """
        获取可上传到SwanLab的指标
        
        Returns:
            metrics: 字典，包含所有可视化指标
        """
        metrics = {}
        
        # 获取专家重要性图
        importance_figs = self.visualize_expert_importance()
        for domain, fig in importance_figs.items():
            metrics[f"Expert_Utilization/{domain}_importance"] = swanlab.Image(fig)
            plt.close(fig)
        
        # 获取专家-类别关联图
        association_figs = self.visualize_expert_class_association()
        for task, fig in association_figs.items():
            metrics[f"Expert_Class_Association/{task}"] = swanlab.Image(fig)
            plt.close(fig)
        
        return metrics
    
    def reset(self):
        """重置收集的数据"""
        self.driver_combine_weights = []
        self.environment_combine_weights = []
        self.driver_expert_class_data = {"emotion": [], "behavior": []}
        self.environment_expert_class_data = {"context": [], "vehicle": []}


# 辅助函数：计算专家利用率的Gini系数
def compute_gini_coefficient(expert_importance):
    """
    计算专家利用率的Gini系数，用于量化专家利用的不平衡程度
    
    Args:
        expert_importance: 专家重要性分数数组
        
    Returns:
        gini: Gini系数 (0表示完全平等，1表示完全不平等)
    """
    # 确保输入是非负的
    if isinstance(expert_importance, torch.Tensor):
        expert_importance = expert_importance.detach().cpu().numpy()
    
    # 排序
    sorted_importance = np.sort(expert_importance)
    n = len(sorted_importance)
    
    # 计算Lorenz曲线下的面积
    cumsum = np.cumsum(sorted_importance)
    # 归一化
    lorenz = cumsum / cumsum[-1]
    
    # 计算面积
    area_under_lorenz = np.sum(lorenz) / n
    
    # Gini系数 = 1 - 2 * 洛伦兹曲线下的面积
    gini = 1 - 2 * area_under_lorenz
    
    return gini

class BSExpertLoss(nn.Module):
    """
    平衡专家损失(Balanced Specialist Expert Loss)
    
    用于处理长尾分布问题，每个专家使用不同的类别先验概率偏置，以平衡不同类别的预测。
    """

    def __init__(self, cls_num_list=None, tau_list=(0, 1, 2), eps=1e-9):
        """
        初始化BSExpertLoss
        
        Args:
            cls_num_list: 每个类别的样本数量列表，用于计算类别先验概率
            tau_list: 每个专家使用的温度参数列表，用于调整类别先验概率的影响
            eps: 小值，用于数值稳定性
        """
        super().__init__()
        self.base_loss = F.cross_entropy

        if cls_num_list is None:
            # 如果未提供类别数量列表，则假设均匀分布
            self.register_buffer('bsce_weight', None)
        else:
            # 将类别数量列表转换为张量并注册为缓冲区
            self.register_buffer('bsce_weight', torch.tensor(cls_num_list).float())
            
        # 将tau_list转换为张量并注册为缓冲区
        self.register_buffer('tau_list', torch.tensor(tau_list).float())
        self.num_experts = len(tau_list)
        self.eps = eps

        assert self.num_experts >= 1

    def forward(self, expert_logits, targets, return_expert_losses=False):
        """
        计算多专家损失
        
        Args:
            expert_logits: 专家输出logits列表，每个元素形状为(batch_size, num_classes)
            targets: 目标标签，形状为(batch_size,)
            return_expert_losses: 是否返回每个专家的损失
            
        Returns:
            loss: 总损失
            expert_losses: 每个专家的损失字典（如果return_expert_losses=True）
        """
        expert_losses = {}
        loss = 0.0

        # 确保expert_logits是列表
        if not isinstance(expert_logits, list):
            expert_logits = [expert_logits]

        # 如果专家数量小于tau_list长度，只使用前n个tau值
        num_actual_experts = len(expert_logits)
        
        for idx in range(num_actual_experts):
            # 获取当前专家的输出
            expert_logit = expert_logits[idx]
            
            # 如果有类别权重，则添加偏置
            if self.bsce_weight is not None and idx < self.num_experts:
                tau = self.tau_list[idx % self.num_experts]
                bias = self.get_default_bias(tau)
                # 确保bias在与expert_logit相同的设备上
                bias = bias.to(expert_logit.device)
                adjusted_expert_logit = expert_logit + bias
            else:
                adjusted_expert_logit = expert_logit
                
            # 计算当前专家的损失
            expert_loss = self.base_loss(adjusted_expert_logit, targets)
            expert_losses[f'loss_e_{idx}'] = expert_loss
            loss = loss + expert_loss

        # 计算平均损失
        loss = loss / num_actual_experts

        if return_expert_losses:
            return loss, expert_losses
        else:
            return loss

    def get_default_bias(self, tau=1):
        """
        计算类别先验概率偏置
        
        Args:
            tau: 温度参数，控制偏置的强度
            
        Returns:
            bias: 类别先验概率偏置，形状为(num_classes,)
        """
        if self.bsce_weight is None:
            return 0.0
            
        prior = self.bsce_weight
        prior = prior / prior.sum()
        log_prior = torch.log(prior + self.eps)
        return tau * log_prior

    def get_bias_from_index(self, e_idx):
        """
        根据专家索引获取偏置
        
        Args:
            e_idx: 专家索引
            
        Returns:
            bias: 类别先验概率偏置
        """
        tau = self.tau_list[e_idx]
        return self.get_default_bias(tau)


class FocalLoss(nn.Module):
    """
    焦点损失(Focal Loss)，用于处理类别不平衡问题
    
    Args:
        cls_num_list: 每个类别的样本数量列表
        weight: 手动指定的类别权重
        gamma: 聚焦参数，控制易分类样本的权重降低程度
    """
    def __init__(self, cls_num_list=None, weight=None, gamma=0.,eps=1e-9):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.eps = eps
        
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float, requires_grad=False)
            self.register_buffer('weight', weight)
        else:
            self.register_buffer('weight', None)
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        """
        计算焦点损失
        
        Args:
            output_logits: 预测logits，形状为(batch_size, num_classes)
            target: 目标标签，形状为(batch_size,)
            
        Returns:
            loss: 焦点损失值
        """
        device = output_logits.device
        weight = self.weight.to(device) if self.weight is not None else None
        
        return self._focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=weight), self.gamma)
    
    def _focal_loss(self, input_values, gamma):
        """
        焦点损失的核心计算
        
        Args:
            input_values: 交叉熵损失值
            gamma: 聚焦参数
            
        Returns:
            loss: 焦点损失值
        """
        p = torch.exp(-input_values)
        p = torch.clamp(p, self.eps, 1. - self.eps)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()


class LDAMLoss(nn.Module):
    """
    标签分布感知边界损失(Label-Distribution-Aware Margin Loss)
    
    为少数类设置更大的分类边界，以处理类别不平衡问题
    
    Args:
        cls_num_list: 每个类别的样本数量列表
        max_m: 最大边界参数
        s: 缩放参数
        reweight_epoch: 开始重新加权的epoch，-1表示不使用重新加权
    """
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # 如果未提供类别数量列表，则不使用LDAM调整
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            # 根据类别频率计算边界值
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.register_buffer('m_list', m_list)  # 使用register_buffer确保m_list会随着模块移动到正确的设备
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # 使用有效样本数计算类别权重
                idx = 1 # 可以根据条件设置idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
                self.register_buffer('per_cls_weights_enabled', per_cls_weights)  # 使用register_buffer
            else:
                self.register_buffer('per_cls_weights_enabled', None)
                self.per_cls_weights = None

    def to(self, device):
        """
        将模块移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 移动后的模块
        """
        super().to(device)
        return self

    def _hook_before_epoch(self, epoch):
        """
        在每个epoch开始前调用，用于更新类别权重
        
        Args:
            epoch: 当前epoch
        """
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        """
        获取添加边界后的最终输出
        
        Args:
            output_logits: 预测logits，形状为(batch_size, num_classes)
            target: 目标标签，形状为(batch_size,)
            
        Returns:
            final_output: 添加边界后的logits
        """
        x = output_logits
        device = x.device  # 获取输入张量的设备

        # 确保m_list在正确的设备上
        m_list = self.m_list.to(device) if self.m_list is not None else None

        # 创建索引张量 - 使用布尔类型而不是uint8
        index = torch.zeros_like(x, dtype=torch.bool, device=device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # 创建one-hot索引
         
        # 将布尔张量转换为浮点数进行矩阵乘法
        index_float = index.float()
        
        # 确保所有张量都在同一设备上
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        # 使用布尔类型的index进行where操作
        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        """
        计算LDAM损失
        
        Args:
            output_logits: 预测logits，形状为(batch_size, num_classes)
            target: 目标标签，形状为(batch_size,)
            
        Returns:
            loss: LDAM损失值
        """
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        device = output_logits.device  # 获取输入张量的设备
        
        # 确保per_cls_weights在正确的设备上
        if self.per_cls_weights is not None:
            per_cls_weights = self.per_cls_weights.to(device)
        else:
            per_cls_weights = None
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=per_cls_weights)
