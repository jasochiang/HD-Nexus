import os
import logging
import torch
import json
import argparse
import math
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradScalerKwargs
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch.nn.functional as F

# 导入自定义的模块和类
from dataset import CarDataset, CarDataCollator
from model.HDmodel import HDMODEL
from aide_metrics import compute_metrics
# from dataset_utils import apply_clip_mixup  # 导入新添加的mixup函数 - 已注释

import optuna.exceptions
from pytorchtools import EarlyStopping

logger = logging.getLogger(__name__)


class LossCalculator:
    """损失计算模块"""

    def __init__(self, loss_type="ce", 
                 focal_gamma=2.0, ldam_max_m=0.5, ldam_s=30, 
                 cls_num_lists=None, reweight_epoch=-1):
        """
        初始化损失计算器
        
        Args:
            loss_type: 单任务损失函数类型 ('ce': 标准交叉熵, 'focal': 焦点损失, 'ldam': LDAM损失)
            focal_gamma: Focal Loss的gamma参数
            ldam_max_m: LDAM Loss的最大边界参数
            ldam_s: LDAM Loss的缩放参数
            cls_num_lists: 各任务类别样本数量列表的字典 {'emotion': [...], 'behavior': [...], ...}
            reweight_epoch: LDAM Loss开始重新加权的epoch
        """
        self.loss_type = loss_type
        
        # 获取类别数量列表
        if cls_num_lists is None:
            # 使用默认的类别分布
            self.cls_num_lists = {
                # Emotion category distribution: Anxiety: 259, Peace: 1123, Weariness: 208, Happiness: 155, Anger: 139
                'emotion': [259, 1123, 208, 155, 139],
                
                # Behavior category distribution: Smoking: 77, Making Phone: 251, Looking Around: 448, 
                # Dozing Off: 16, Normal Driving: 836, Talking: 94, Body Movement: 162
                'behavior': [77, 251, 448, 16, 836, 94, 162],
                
                # Context category distribution: Traffic Jam: 226, Waiting: 430, Smooth Traffic: 1228
                'context': [226, 430, 1228],
                
                # Vehicle category distribution: Parking: 473, Turning: 221, Backward Moving: 114, 
                # Changing Lane: 81, Forward Moving: 995
                'vehicle': [473, 221, 114, 81, 995]
            }
        else:
            self.cls_num_lists = cls_num_lists
        
        # 根据loss_type初始化损失函数
        if loss_type == "focal":
            from model.utils import FocalLoss
            self.emotion_loss_fn = FocalLoss(cls_num_list=self.cls_num_lists['emotion'], gamma=focal_gamma)
            self.behavior_loss_fn = FocalLoss(cls_num_list=self.cls_num_lists['behavior'], gamma=focal_gamma)
            self.context_loss_fn = FocalLoss(cls_num_list=self.cls_num_lists['context'], gamma=focal_gamma)
            self.vehicle_loss_fn = FocalLoss(cls_num_list=self.cls_num_lists['vehicle'], gamma=focal_gamma)
        elif loss_type == "ldam":
            from model.utils import LDAMLoss
            self.emotion_loss_fn = LDAMLoss(cls_num_list=self.cls_num_lists['emotion'], 
                                           max_m=ldam_max_m, s=ldam_s, reweight_epoch=reweight_epoch)
            self.behavior_loss_fn = LDAMLoss(cls_num_list=self.cls_num_lists['behavior'], 
                                            max_m=ldam_max_m, s=ldam_s, reweight_epoch=reweight_epoch)
            self.context_loss_fn = LDAMLoss(cls_num_list=self.cls_num_lists['context'], 
                                           max_m=ldam_max_m, s=ldam_s, reweight_epoch=reweight_epoch)
            self.vehicle_loss_fn = LDAMLoss(cls_num_list=self.cls_num_lists['vehicle'], 
                                           max_m=ldam_max_m, s=ldam_s, reweight_epoch=reweight_epoch)
        else:  # 默认使用标准交叉熵损失
            self.emotion_loss_fn = F.cross_entropy
            self.behavior_loss_fn = F.cross_entropy
            self.context_loss_fn = F.cross_entropy
            self.vehicle_loss_fn = F.cross_entropy

    def _hook_before_epoch(self, epoch):
        """
        在每个epoch开始前调用，用于更新损失函数的状态
        
        Args:
            epoch: 当前epoch
        """
        if self.loss_type == "ldam":
            self.emotion_loss_fn._hook_before_epoch(epoch)
            self.behavior_loss_fn._hook_before_epoch(epoch)
            self.context_loss_fn._hook_before_epoch(epoch)
            self.vehicle_loss_fn._hook_before_epoch(epoch)

    def compute_task_losses(self, outputs, emotion_labels, behavior_labels, context_labels, vehicle_labels):
        """计算各任务的损失"""
        emotion_output, behavior_output, context_output, vehicle_output = outputs

        # 根据loss_type使用相应的损失函数
        if self.loss_type == "focal" or self.loss_type == "ldam":
            loss_emotion = self.emotion_loss_fn(emotion_output, emotion_labels)
            loss_behavior = self.behavior_loss_fn(behavior_output, behavior_labels)
            loss_context = self.context_loss_fn(context_output, context_labels)
            loss_vehicle = self.vehicle_loss_fn(vehicle_output, vehicle_labels)
        else:  # 默认使用标准交叉熵损失
            loss_emotion = F.cross_entropy(emotion_output, emotion_labels)
            loss_behavior = F.cross_entropy(behavior_output, behavior_labels)
            loss_context = F.cross_entropy(context_output, context_labels)
            loss_vehicle = F.cross_entropy(vehicle_output, vehicle_labels)

        return [loss_emotion, loss_behavior, loss_context, loss_vehicle]

    def compute_total_dwa_loss(self, task_losses, task_weights):
        """计算DWA加权总损失"""
        total_loss = sum(w * l for w, l in zip(task_weights, task_losses))

        return total_loss

    def get_loss_items(self, task_losses):
        """获取损失的标量值"""
        return [l.item() for l in task_losses]


class DWAManager:
    """动态权重调整（DWA）管理模块"""

    def __init__(self, temperature=1.0, num_tasks=4):
        self.temperature = temperature
        self.num_tasks = num_tasks
        self.previous_epoch_losses = []
        self.task_weights = [1.0] * num_tasks

    def update_weights(self, current_epoch_losses, device):
        """更新任务权重"""
        if not current_epoch_losses:
            return self.task_weights

        # 计算当前epoch的平均损失
        avg_losses = torch.tensor(current_epoch_losses, device=device).mean(dim=0)
        self.previous_epoch_losses.append(avg_losses)

        # 如果有至少两个epoch的损失历史，计算DWA权重
        if len(self.previous_epoch_losses) >= 2:
            last = self.previous_epoch_losses[-1]
            second_last = self.previous_epoch_losses[-2]

            # 计算损失比率
            w = [last[i] / second_last[i] if second_last[i] != 0 else 1.0
                 for i in range(len(second_last))]
            w = torch.tensor(w, device=device)

            # 应用softmax和缩放
            self.task_weights = (F.softmax(w / self.temperature, dim=0) * self.num_tasks).tolist()

        return self.task_weights

    def get_weights(self):
        """获取当前任务权重"""
        return self.task_weights


class ModelManager:
    """模型保存和加载管理模块"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_metrics = {}  # 存储每个指标的最佳值

    def is_best_metric(self, metric_name, metric_value, higher_is_better=True):
        """判断是否是最佳指标"""
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = metric_value
            return True

        current_best = self.best_metrics[metric_name]
        if higher_is_better:
            is_better = metric_value > current_best
        else:
            is_better = metric_value < current_best

        if is_better:
            self.best_metrics[metric_name] = metric_value
            return True
        return False

    def save_best_model(self, accelerator, model, metric_name, metric_value, task_weights=None, higher_is_better=True):
        """保存最佳模型（如果当前指标是最佳的话）"""
        # 检查是否是最佳指标
        if not self.is_best_metric(metric_name, metric_value, higher_is_better):
            return False

        # 确定保存目录 - 使用指标名称
        save_dir = os.path.join(self.output_dir, f"best_{metric_name}_model")
        os.makedirs(save_dir, exist_ok=True)

        # 解包 & 保存权重
        unwrapped_model = accelerator.unwrap_model(model)  # 关键一步！
        torch.save(unwrapped_model.state_dict(),os.path.join(save_dir, "pytorch_model.bin"))
        logger.info(f"已保存最佳 {metric_name} 模型权重，{metric_name}: {metric_value}")

        """
        # 保存模型状态
        accelerator.save_state(save_dir)
        logger.info(f"已保存最佳 {metric_name} 模型，{metric_name}: {metric_value}")
        """

        # 保存DWA权重
        if accelerator.is_main_process and task_weights is not None:
            with open(os.path.join(save_dir, 'task_weights.json'), 'w') as f:
                json.dump(task_weights, f)

        return True

    def load_model_and_weights(self, accelerator, model, metric_name):
        """根据指标名称加载模型和DWA权重"""
        # 确定加载目录
        model_path = os.path.join(self.output_dir, f"best_{metric_name}_model")

        if not os.path.exists(model_path):
            logger.warning(f"未找到 {metric_name} 指标对应的模型: {model_path}")
            return None
        """
        # 加载模型状态
        accelerator.wait_for_everyone()
        accelerator.load_state(model_path)
        logger.info(f"已从 {model_path} 加载最佳 {metric_name} 模型")
        """

        try:
            # 尝试加载pytorch_model.bin
            ckpt_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.isfile(ckpt_file):
                # 同步加载
                state_dict = torch.load(ckpt_file, map_location="cpu")
                accelerator.unwrap_model(model).load_state_dict(state_dict, strict=True)
                logger.info(f"已从 {ckpt_file} 恢复权重")
            else:
                # 如果找不到pytorch_model.bin，尝试加载其他格式的权重文件
                alternative_files = ["model.safetensors", "optimizer.bin", "scheduler.bin", "scaler.pt", "random_states_0.pkl"]
                loaded = False
                
                for alt_file in alternative_files:
                    alt_path = os.path.join(model_path, alt_file)
                    if os.path.isfile(alt_path):
                        logger.info(f"尝试从 {alt_path} 加载权重")
                        # 使用accelerator直接加载状态
                        accelerator.load_state(model_path)
                        logger.info(f"已使用accelerator从 {model_path} 加载模型状态")
                        loaded = True
                        break
                
                if not loaded:
                    logger.warning(f"未找到任何可用的权重文件，无法加载模型")
                    return None
        except Exception as e:
            logger.error(f"加载模型权重时出错: {str(e)}")
            try:
                # 如果加载出错，尝试使用accelerator直接加载
                accelerator.load_state(model_path)
                logger.info(f"已使用accelerator从 {model_path} 加载模型状态")
            except Exception as e2:
                logger.error(f"使用accelerator加载模型状态时也出错: {str(e2)}")
                return None

        accelerator.wait_for_everyone()  # ★ 多卡时保持同步

        # 加载DWA权重
        task_weights_path = os.path.join(model_path, 'task_weights.json')
        if os.path.exists(task_weights_path):
            with open(task_weights_path, 'r') as f:
                task_weights = json.load(f)
            logger.info(f"已从 {task_weights_path} 加载DWA权重: {task_weights}")
        else:
            task_weights = [1.0] * 4
            logger.warning("未找到DWA权重文件，使用默认权重 [1.0, 1.0, 1.0, 1.0]")

        return task_weights
def save_test_results(test_metrics, output_dir, model_type_name):
    if not test_metrics:
        return
    test_metrics_to_save = {k: v for k, v in test_metrics.items() if k != 'confusion_matrices'}
    filename = f'test_metrics_{model_type_name}.json'
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(test_metrics_to_save, f, indent=2)
    logger.info(f"测试指标已保存到 {filename}")

def evaluate(model, dataloader, accelerator, task_weights, analyze_experts=True, is_test_phase=False):
    """
    在分布式环境中评估模型。
    
    Args:
        model: 要评估的模型
        dataloader: 评估数据加载器
        accelerator: Accelerator实例
        task_weights: 任务权重列表
        analyze_experts: 是否分析专家利用率（只在验证阶段设为True）
        
    Returns:
        评估指标字典和混淆矩阵图
    """
    model.eval()
    all_logits = []
    all_labels_list = []
    
    expert_analyzer = None
    if analyze_experts:
        from model.utils import ExpertUtilizationAnalyzer
        expert_analyzer = ExpertUtilizationAnalyzer()
    
    # 初始化损失计算器 - 在验证和测试阶段只使用CE损失
    loss_calculator = LossCalculator(
        loss_type="ce"  # 在验证和测试阶段固定使用CE损失
    )
    
    progress_bar = tqdm(
        dataloader, 
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )

    # 确保所有进程同步开始
    accelerator.wait_for_everyone()

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):

            # 提取标签
            emotion_labels = batch.pop("emotion_label")
            behavior_labels = batch.pop("behavior_label")
            context_labels = batch.pop("context_label")
            vehicle_labels = batch.pop("vehicle_label")
            
            # 前向传播
            extra_returns = {}
            
            model_kwargs = batch.copy()

            # 模型前向传播
            raw_outputs = model(**model_kwargs)

            # 解包模型输出
            if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
                outputs, extra_returns = raw_outputs
            else:
                outputs = raw_outputs
            
            emotion_output, behavior_output, context_output, vehicle_output = outputs

            if analyze_experts and expert_analyzer is not None and 'combine_weights' in extra_returns:
                combine_weights_dict = extra_returns['combine_weights']
                expert_analyzer.collect_combine_weights(combine_weights_dict, domain="driver")
                expert_analyzer.collect_combine_weights(combine_weights_dict, domain="environment")
                expert_analyzer.collect_expert_class_data(combine_weights_dict, emotion_labels, domain="driver", task="emotion")
                expert_analyzer.collect_expert_class_data(combine_weights_dict, behavior_labels, domain="driver", task="behavior")
                expert_analyzer.collect_expert_class_data(combine_weights_dict, context_labels, domain="environment", task="context")
                expert_analyzer.collect_expert_class_data(combine_weights_dict, vehicle_labels, domain="environment", task="vehicle")

            # 准备标签元组
            labels_tuple = (emotion_labels, behavior_labels, context_labels, vehicle_labels)
            logits_tuple = (emotion_output, behavior_output, context_output, vehicle_output)
            
            # 收集所有进程的预测和标签
            gathered_logits = accelerator.gather_for_metrics(logits_tuple)
            gathered_labels = accelerator.gather_for_metrics(labels_tuple)

            all_logits.append(gathered_logits)
            all_labels_list.append(gathered_labels)

    metrics = None
    correctly_predicted_samples = None

    # 仅在主进程中计算指标
    if accelerator.is_main_process:
        # 合并所有批次的logits和labels
        emotion_logits, behavior_logits, context_logits, vehicle_logits = zip(*all_logits)
        emotion_labels_all, behavior_labels_all, context_labels_all, vehicle_labels_all = zip(*all_labels_list)

        # 将列表中的张量连接起来
        emotion_logits = torch.cat(emotion_logits, dim=0)
        behavior_logits = torch.cat(behavior_logits, dim=0)
        context_logits = torch.cat(context_logits, dim=0)
        vehicle_logits = torch.cat(vehicle_logits, dim=0)

        emotion_labels_all = torch.cat(emotion_labels_all, dim=0)
        behavior_labels_all = torch.cat(behavior_labels_all, dim=0)
        context_labels_all = torch.cat(context_labels_all, dim=0)
        vehicle_labels_all = torch.cat(vehicle_labels_all, dim=0)

        # 计算评估指标
        eval_metrics = compute_metrics((
            (emotion_logits.cpu(), behavior_logits.cpu(), context_logits.cpu(), vehicle_logits.cpu()),
            (emotion_labels_all.cpu(), behavior_labels_all.cpu(), context_labels_all.cpu(), vehicle_labels_all.cpu())
        ))

        # 如果是测试阶段，找出所有任务都预测正确的样本
        if is_test_phase:
            emotion_preds = torch.argmax(emotion_logits, dim=1).cpu()
            behavior_preds = torch.argmax(behavior_logits, dim=1).cpu()
            context_preds = torch.argmax(context_logits, dim=1).cpu()
            vehicle_preds = torch.argmax(vehicle_logits, dim=1).cpu()

            correct_emotion = (emotion_preds == emotion_labels_all.cpu())
            correct_behavior = (behavior_preds == behavior_labels_all.cpu())
            correct_context = (context_preds == context_labels_all.cpu())
            correct_vehicle = (vehicle_preds == vehicle_labels_all.cpu())

            all_correct_mask = correct_emotion & correct_behavior & correct_context & correct_vehicle
            
            correct_indices = torch.where(all_correct_mask)[0]

            # 筛选并选择高置信度样本
            correctly_predicted_samples = []
            high_confidence_samples = []
            
            for idx in correct_indices:
                emotion_label = emotion_labels_all[idx].item()
                behavior_label = behavior_labels_all[idx].item()
                
                # 过滤掉 Peace (标签1) 和 Normal Driving (标签4)
                if emotion_label == 1 or behavior_label == 4:  # Peace=1, Normal Driving=4
                    continue
                
                # 计算各任务的置信度（使用softmax后的最大概率）
                emotion_confidence = torch.softmax(emotion_logits[idx], dim=0).max().item()
                behavior_confidence = torch.softmax(behavior_logits[idx], dim=0).max().item()
                context_confidence = torch.softmax(context_logits[idx], dim=0).max().item()
                vehicle_confidence = torch.softmax(vehicle_logits[idx], dim=0).max().item()
                
                # 计算平均置信度
                avg_confidence = (emotion_confidence + behavior_confidence + context_confidence + vehicle_confidence) / 4.0
                
                sample_data = {
                    "sample_index": idx.item(),
                    "emotion_label": emotion_label,
                    "behavior_label": behavior_label,
                    "context_label": context_labels_all[idx].item(),
                    "vehicle_label": vehicle_labels_all[idx].item(),
                    "emotion_confidence": emotion_confidence,
                    "behavior_confidence": behavior_confidence,
                    "context_confidence": context_confidence,
                    "vehicle_confidence": vehicle_confidence,
                    "avg_confidence": avg_confidence
                }
                high_confidence_samples.append(sample_data)
            
            # 按平均置信度排序，选择前10个
            high_confidence_samples.sort(key=lambda x: x['avg_confidence'], reverse=True)
            correctly_predicted_samples = high_confidence_samples[:10]
            
            if accelerator.is_main_process:
                logger.info(f"筛选后的高置信度样本数量: {len(correctly_predicted_samples)}")
                if correctly_predicted_samples:
                    logger.info(f"最高置信度: {correctly_predicted_samples[0]['avg_confidence']:.4f}")
                    logger.info(f"最低置信度: {correctly_predicted_samples[-1]['avg_confidence']:.4f}")
                else:
                    logger.info("未找到符合条件的样本（排除Peace和Normal Driving后）")

        # 计算验证集损失
        outputs = (emotion_logits, behavior_logits, context_logits, vehicle_logits)
        task_losses = loss_calculator.compute_task_losses(
            outputs, emotion_labels_all, behavior_labels_all, context_labels_all, vehicle_labels_all
        )
        
        total_loss = loss_calculator.compute_total_dwa_loss(task_losses, task_weights)

        eval_metrics['loss'] = total_loss.item()
        eval_metrics['emotion_loss'] = task_losses[0].item()
        eval_metrics['behavior_loss'] = task_losses[1].item()
        eval_metrics['context_loss'] = task_losses[2].item()
        eval_metrics['vehicle_loss'] = task_losses[3].item()
        
        if analyze_experts and expert_analyzer is not None:
            # 计算并添加专家利用率指标
            driver_importance, environment_importance = expert_analyzer.compute_expert_importance()
            
            # 计算Gini系数
            from model.utils import compute_gini_coefficient
            if driver_importance is not None:
                driver_gini = compute_gini_coefficient(driver_importance)
                eval_metrics['driver_expert_gini'] = driver_gini
                
            if environment_importance is not None:
                environment_gini = compute_gini_coefficient(environment_importance)
                eval_metrics['environment_expert_gini'] = environment_gini

        metrics = eval_metrics
    
    # 确保所有进程在评估后同步
    accelerator.wait_for_everyone()
    
    if is_test_phase:
        return metrics, correctly_predicted_samples
    else:
        return metrics

def setup_environment(args, optuna_accelerator=None):
    """环境与配置初始化模块"""
    # 创建训练配置字典，直接使用命令行参数
    train_config = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "metric_for_best_model": args.metric_for_best_model,
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
        "fp16": args.fp16,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "loss_type": args.loss_type,
        "early_stopping_patience": getattr(args, 'early_stopping_patience', 7),  # 新增：早停耐心值
        "use_differential_lr": getattr(args, 'use_differential_lr', False),  # 新增：差异化学习率标志
        "soft_num_experts": getattr(args, 'soft_num_experts', 8),  # 新增：软专家数量
        "mi_loss_alpha": getattr(args, 'mi_loss_alpha', 0.1),  # 新增：互信息损失权重参数
        # "mixup_alpha": getattr(args, 'mixup_alpha', 0.3), # 新增：Mixup插值参数 - 已注释
    }

    # 创建输出目录
    os.makedirs(train_config["output_dir"], exist_ok=True)
    
    # 检查输出目录是否已存在且不为空
    if (
        os.path.exists(train_config["output_dir"]) 
        and os.listdir(train_config["output_dir"])
        and not getattr(args, 'overwrite_output_dir', False)
        and not getattr(args, 'eval_only', False)
    ):
        raise ValueError(
            f"输出目录 ({train_config['output_dir']}) 已存在且不为空。使用 --overwrite_output_dir 参数覆盖。"
        )
    
    # 从配置中读取加速器相关参数
    accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    use_fp16 = train_config.get("fp16", False)
    mixed_precision = "fp16" if use_fp16 else "no"

    # 使用传入的accelerator或创建新的
    if optuna_accelerator is not None:
        accelerator = optuna_accelerator
        logger.info("使用已存在的Accelerator实例")
    else:
        # 创建新的accelerator实例
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        grad_kwargs = GradScalerKwargs(init_scale=2 ** 16)  # 自定义 loss-scale 的初始值
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=accumulation_steps,
            kwargs_handlers=[grad_kwargs,ddp_kwargs],
        )
        logger.info("创建了新的Accelerator实例")

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.process_index in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "进程序号: %s, 设备: %s, n_gpu: %s, 分布式训练: %s, 16位训练: %s",
        accelerator.process_index,
        accelerator.device,
        accelerator.num_processes,
        accelerator.distributed_type != "NO",
        use_fp16,
    )
    logger.info("训练参数 %s", train_config)
    logger.info("命令行参数 %s", args)
    
    return accelerator, train_config

def prepare_data(train_config):
    """数据准备模块"""
    with open("keypoints_stats.json", "r") as f:
        stats = json.load(f)
    
    # 实例化数据集和数据整理器
    train_dataset = CarDataset(csv_file='/media/Data1/aide/training.csv', mode='train', state=stats)
    val_dataset = CarDataset(csv_file='/media/Data1/aide/validation.csv', mode=None, state=stats)
    test_dataset = CarDataset(csv_file='/media/Data1/aide/testing.csv', mode=None, state=stats)
    
    # 获取mixup_alpha参数，只在训练时应用mixup
    # mixup_alpha = train_config.get("mixup_alpha", 0.0)
    # logger.info(f"Mixup alpha: {mixup_alpha}")
    
    # 创建数据整理器，传入mixup_alpha参数
    # train_data_collator = CarDataCollator(mixup_alpha=mixup_alpha)
    # eval_data_collator = CarDataCollator(mixup_alpha=0.0)  # 评估和测试时不使用mixup
    data_collator = CarDataCollator()
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.get("per_device_train_batch_size", 8),
        shuffle=True,
        num_workers=train_config.get("dataloader_num_workers", 4),
        collate_fn=data_collator,
        pin_memory=False,
        drop_last=False
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.get("per_device_eval_batch_size", 8),
        shuffle=False,
        num_workers=train_config.get("dataloader_num_workers", 4),
        collate_fn=data_collator,
        pin_memory=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_config.get("per_device_eval_batch_size", 8),
        shuffle=False,
        num_workers=train_config.get("dataloader_num_workers", 4),
        collate_fn=data_collator,
        pin_memory=False
    )
    
    return train_dataloader, eval_dataloader, test_dataloader

def prepare_model_and_optimizer(train_config, train_dataloader):
    """模型与优化器准备模块"""
    soft_num_experts = train_config.get("soft_num_experts", 8)
    mi_loss_alpha = train_config.get("mi_loss_alpha", 0.1)
    logger.info(f"软专家数量: {soft_num_experts}")
    model = HDMODEL(mi_loss_alpha=mi_loss_alpha, soft_num_experts=soft_num_experts)
    logger.info("成功实例化模型: HDMODEL")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info("Converted model to SyncBatchNorm")

    use_diff_lr = train_config.get("use_differential_lr", False)
    base_lr = float(train_config.get("learning_rate", 1e-3))
    
    if use_diff_lr:
        param_groups = []
        for name, module in model.named_modules():
            if hasattr(module, 'get_optimizer_parameters'):
                logger.info(f"为模块 {name} 应用差分学习率")
                param_groups.extend(module.get_optimizer_parameters(head_lr=base_lr))
        
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and not any(param is p for group in param_groups for p in group['params']):
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr
            })
            logger.info(f"为其他参数应用基础学习率: {base_lr}")
        
        optimizer = AdamW(
            param_groups,
            weight_decay=float(train_config.get("weight_decay", 1e-4))
        )
        
        for i, group in enumerate(param_groups):
            logger.info(f"参数组 {i+1}: lr = {group['lr']}, 参数数量 = {len(group['params'])}")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=float(train_config.get("weight_decay", 1e-4))
        )

    accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    num_training_steps = train_config.get("num_train_epochs", 40) * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=train_config.get("lr_scheduler_type", "linear"),
        optimizer=optimizer,
        num_warmup_steps=train_config.get("num_warmup_steps", 0),
        num_training_steps=num_training_steps
    )
    
    return model, optimizer, lr_scheduler

def run_training_loop(accelerator, train_config, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, mi_loss_alpha, trial: 'optuna.trial.Trial' = None):
    """核心训练循环模块 - 重构为模块化架构"""

    # 初始化模块化组件
    loss_calculator = LossCalculator(
        loss_type=train_config.get("loss_type", "ce"),
        focal_gamma=train_config.get("focal_gamma", 2.0),
        ldam_max_m=train_config.get("ldam_max_m", 0.5),
        ldam_s=train_config.get("ldam_s", 30.0),
        reweight_epoch=train_config.get("reweight_epoch", -1)
    )
    dwa_manager = DWAManager(temperature=1.0, num_tasks=4)
    task_weights = dwa_manager.get_weights()
    
    model_manager = ModelManager(train_config['output_dir'])

    # 初始化追踪变量
    best_metric = 0.0
    # 初始化早停信号
    earlystop_signal = torch.tensor(0.0, device=accelerator.device)

    # 获取用于选择最佳模型的指标
    metric_for_best_model = train_config.get("metric_for_best_model", "All_F1")
    if accelerator.is_main_process:
        logger.info(f"使用 {metric_for_best_model} 作为选择最佳模型的指标")

    # 初始化早停机制
    early_stopping_patience = train_config.get("early_stopping_patience", 7)
    early_stopping_path = os.path.join(train_config['output_dir'], 'early_stopping_checkpoint.pt')
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True,
        path=early_stopping_path,
        trace_func=logger.info if accelerator.is_main_process else lambda x: None
    )

    if accelerator.is_main_process:
        logger.info(f"早停机制已启用，耐心值: {early_stopping_patience}, 基于指标: {metric_for_best_model}")

    # 训练循环
    num_epochs = train_config.get("num_train_epochs", 40)

    # tqdm可以迭代显示进度条,disable=not accelerator.is_main_process 表示只在主进程显示进度条
    for epoch in tqdm(range(num_epochs), desc="Epochs", disable=not accelerator.is_main_process):
        # 在每个epoch开始前更新损失函数状态
        loss_calculator._hook_before_epoch(epoch)

        # === 训练阶段 ===
        model.train()
        current_epoch_losses_list = []

        # 初始化epoch统计
        epoch_stats = {
            'total_loss': 0.0,
            'task_losses': [0.0, 0.0, 0.0, 0.0],  # [emotion, behavior, context, vehicle]
            'mi_loss': 0.0,
            'num_batches': 0
        }

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Training Epoch {epoch}",
            disable=not accelerator.is_main_process
        )

        # 训练数据集迭代
        for step, batch in enumerate(progress_bar):
            # 通过梯度累积，可以多次前向和反向传播后再更新一次参数，相当于模拟更大的 batch size。
            # accelerator.accumulate(model) 是一个上下文管理器，只有在累积步数达到设定值时才会执行 optimizer.step() 和梯度清零。
            # 这样可以有效利用有限的显存资源，同时提升训练效果。
            with accelerator.accumulate(model):
                # 提取标签
                # 在实际调用模型前，我们已经用.pop()方法将标签从batch字典中移除了
                emotion_labels = batch.pop("emotion_label")
                behavior_labels = batch.pop("behavior_label")
                context_labels = batch.pop("context_label")
                vehicle_labels = batch.pop("vehicle_label")

                # 前向传播
                # **batch 表示将batch字典中的所有键值对作为关键字参数传递给模型，例如posture=batch["posture"]
                mi_loss = None
                raw_outputs = model(**batch)
                if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
                    outputs, extra_returns = raw_outputs
                    mi_loss = extra_returns.get("mi_loss")
                else:
                    outputs = raw_outputs

                task_losses = loss_calculator.compute_task_losses(
                    outputs, emotion_labels, behavior_labels, context_labels, vehicle_labels
                )

                # 收集损失用于DWA
                current_epoch_losses_list.append(loss_calculator.get_loss_items(task_losses))

                # 计算总损失
                total_loss = loss_calculator.compute_total_dwa_loss(task_losses, task_weights)
                if mi_loss is not None:
                    total_loss += mi_loss_alpha * mi_loss

                # 反向传播
                accelerator.backward(total_loss)

                # 梯度裁剪
                if accelerator.sync_gradients:
                    if train_config.get("max_grad_norm", 0) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), train_config.get("max_grad_norm"))

                    # 优化器步进
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()

                # 统计损失
                epoch_stats['total_loss'] += total_loss.item()
                for i, task_loss in enumerate(task_losses):
                    epoch_stats['task_losses'][i] += task_loss.item()
                if mi_loss is not None:
                    epoch_stats['mi_loss'] += mi_loss.item()

                epoch_stats['num_batches'] += 1

                # 更新进度条
                progress_bar.set_postfix({
                    "loss": total_loss.item(),
                    "emotion_loss": task_losses[0].item(),
                    "behavior_loss": task_losses[1].item(),
                    "context_loss": task_losses[2].item(),
                    "vehicle_loss": task_losses[3].item()
                })

        # 计算平均损失
        num_batches = max(epoch_stats['num_batches'], 1)
        avg_epoch_loss = epoch_stats['total_loss'] / num_batches

        # === 评估阶段 ===
        eval_metrics = evaluate(
            model, 
            eval_dataloader, 
            accelerator, 
            task_weights, 
            analyze_experts=True,
        )

        # 初始化剪枝信号
        stop_signal = torch.tensor(0.0, device=accelerator.device)


        # === 主进程处理指标和日志 ===
        if accelerator.is_main_process:
            task_weights = dwa_manager.update_weights(current_epoch_losses_list, accelerator.device)

            # 处理评估指标
            if eval_metrics:
                logger.info(f"Epoch {epoch}: {eval_metrics}")

                # Optuna剪枝检查
                if trial is not None:
                    # 获取用于优化的核心指标值
                    metric_value = eval_metrics.get(metric_for_best_model, 0.0)

                    # 向Optuna报告当前Epoch的性能
                    trial.report(metric_value, epoch)

                    # 检查是否需要剪枝
                    if trial.should_prune():
                        logger.info(f"Trial被剪枝在epoch {epoch}，当前{metric_for_best_model}: {metric_value}")
                        # 设置剪枝信号
                        stop_signal.fill_(1.0)

                logger.info(
                    f"Epoch {epoch} train_loss={avg_epoch_loss:.4f} "
                    f"val_loss={eval_metrics.get('loss', 0.0):.4f} "
                    f"all_f1={eval_metrics.get('All_F1', 0.0):.4f} "
                    f"mAcc={eval_metrics.get('mAcc', 0.0):.4f}"
                )

        # === 分布式剪枝同步 ===
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1:
            if accelerator.is_main_process:
                logger.info(f"Trial被剪枝在epoch {epoch}，通知所有进程一同退出。")
            # 关键：让所有进程都抛出这个异常，以确保它们都以相同的方式退出训练循环
            raise optuna.exceptions.TrialPruned()

        # === 模型保存 - 使用新的ModelManager API ===
        if accelerator.is_main_process and eval_metrics is not None:
            # 保存最佳指标模型
            # 定义需要保存的模型配置
            models_to_save = [
                (metric_for_best_model, eval_metrics[metric_for_best_model], True),
                ("loss", eval_metrics['loss'], False),
                ("mAcc", eval_metrics['mAcc'], True)
            ]

            for metric_name, metric_value, higher_is_better in models_to_save:
                model_manager.save_best_model(
                    accelerator, model, metric_name, metric_value,
                    task_weights=task_weights, higher_is_better=higher_is_better
                )

            current_metric_value = eval_metrics['loss']
            val_loss_for_early_stopping = current_metric_value
            early_stopping(val_loss_for_early_stopping, model)

            if early_stopping.early_stop:
                earlystop_signal.fill_(1.0)

        # === 分布式剪枝同步 ===
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(earlystop_signal, src=0)

        if earlystop_signal.item() == 1:
            if accelerator.is_main_process:
                logger.info(f"早停触发！最佳 {metric_for_best_model}: {-early_stopping.val_loss_min:.6f}")
            break

        accelerator.wait_for_everyone()

    return best_metric

def run_final_evaluation(accelerator, train_config, model, test_dataloader):
    """最终测试模块 - 重构为模块化架构"""

    # 初始化模块化组件
    model_manager = ModelManager(train_config['output_dir'])

    # 获取用于训练的主要指标名称
    metric_for_best_model = train_config.get("metric_for_best_model", "All_F1")
    
    # 定义要评估的模型配置
    models_to_evaluate = [
        {
            "name": metric_for_best_model,
            "display_name": f"最佳 {metric_for_best_model}",
            "file_suffix": f"best_{metric_for_best_model}",
        },
        {
            "name": "loss",
            "display_name": "最低loss",
            "file_suffix": "best_loss",
        },
        {
            "name": "mAcc",
            "display_name": "最佳mAcc",
            "file_suffix": "best_mAcc",
        }

    ]

    # 循环评估所有配置的模型
    for model_config in models_to_evaluate:
        if accelerator.is_main_process:
            logger.info(f"开始评估{model_config['display_name']}模型")
        task_weights = model_manager.load_model_and_weights(accelerator, model, metric_name=model_config['name'])

        if task_weights is not None:
            test_metrics, correctly_predicted_samples = evaluate(
                model, 
                test_dataloader, 
                accelerator, 
                task_weights, 
                analyze_experts=False, # 测试阶段不进行gini等分析
                is_test_phase=True
            )

            # 处理测试结果
            if accelerator.is_main_process and test_metrics is not None:
                logger.info(f"{model_config['display_name']}模型测试结果: {test_metrics}")

                save_test_results(test_metrics, train_config['output_dir'], model_config['file_suffix'])

                # 保存完全预测正确的样本信息
                if correctly_predicted_samples is not None:
                    correct_samples_filename = f"high_confidence_correct_predictions_{model_config['file_suffix']}.json"
                    correct_samples_path = os.path.join(train_config['output_dir'], correct_samples_filename)
                    with open(correct_samples_path, 'w') as f:
                        json.dump(correctly_predicted_samples, f, indent=2)
                    logger.info(f"高置信度完全预测正确样本信息已保存到: {correct_samples_path}")
                    logger.info(f"样本数量: {len(correctly_predicted_samples)}")
                    if correctly_predicted_samples:
                        logger.info(f"平均置信度范围: {correctly_predicted_samples[-1]['avg_confidence']:.4f} - {correctly_predicted_samples[0]['avg_confidence']:.4f}")

                

def main():
    # 使用argparse处理命令行参数
    parser = argparse.ArgumentParser(description="训练和评估多模态模型")
    # 基本配置
    parser.add_argument("--output_dir", type=str, default="/media/Data3/xxx", help="输出目录，覆盖配置文件中的设置")
    parser.add_argument("--checkpoint_path", type=str, help="模型检查点路径")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="如果设置，将覆盖输出目录中的现有内容")
    parser.add_argument("--eval_only", action="store_true", help="如果设置，将只进行评估而不进行训练")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal", "ldam"], help="单任务损失函数类型(ce: 标准交叉熵, focal: 焦点损失, ldam: LDAM损失)")
    parser.add_argument("--mi_loss_alpha", type=float, default=0.02, help="MI损失权重参数（HDMODEL生效）")
    parser.add_argument("--use_differential_lr", action="store_true", help="是否使用差分学习率")
    # parser.add_argument("--mixup_alpha", type=float, default=0.3, help="Mixup插值参数alpha，设为0禁用Mixup")
    
    # 损失函数特定参数
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss的gamma参数，控制难易样本权重")
    parser.add_argument("--ldam_max_m", type=float, default=0.5, help="LDAM Loss的最大边界参数")
    parser.add_argument("--ldam_s", type=float, default=30.0, help="LDAM Loss的缩放参数")
    parser.add_argument("--reweight_epoch", type=int, default=10, help="LDAM Loss开始重新加权的epoch")

    # 模型配置
    parser.add_argument("--num_train_epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="每个设备的训练批量大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每个设备的评估批量大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="保存策略")
    parser.add_argument("--save_total_limit", type=int, default=1, help="保存的模型总数限制")
    parser.add_argument("--metric_for_best_model", type=str, default="All_F1", help="用于选择最佳模型的指标")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="数据加载的工作线程数")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子")
    parser.add_argument("--fp16", action="store_true", default=False, help="是否使用16位精度")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="学习率调度器类型")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="预热步数")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停耐心值，连续多少个epoch验证指标不改善就停止训练")
    parser.add_argument("--soft_num_experts", type=int, default=8, help="软专家数量")
    args = parser.parse_args()

    # 1. 环境与配置初始化
    accelerator, train_config = setup_environment(args)

    # 2. 数据准备
    train_dataloader, eval_dataloader, test_dataloader = prepare_data(train_config)

    # 3. 模型与优化器准备
    model, optimizer, lr_scheduler = prepare_model_and_optimizer(train_config, train_dataloader)

    # 4. 使用accelerator.prepare包装所有组件
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    if args.checkpoint_path is not None:
        logger.info(f"检测到检查点路径: {args.checkpoint_path}")
        accelerator.load_state(args.checkpoint_path)
        logger.info(f"已从 {args.checkpoint_path} 加载检查点")

    # 根据eval_only参数决定是否进行训练
    if not args.eval_only:
        logger.info("开始训练过程...")
        run_training_loop(
            accelerator, train_config, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, args.mi_loss_alpha
        )
    else:
        logger.info("跳过训练过程，直接进行评估...")

    # 训练结束，在测试集上评估
    run_final_evaluation(accelerator, train_config, model, test_dataloader)
    
    # 最终同步，确保所有进程一起退出
    accelerator.wait_for_everyone()

    del model, optimizer, lr_scheduler  # 若变量名不同请替换
    torch.cuda.empty_cache()

    # 一键收尾：barrier + stop trackers + destroy PG
    accelerator.end_training()

    logger.info("训练完成")


def run_trial(trial, args, accelerator):
    # 1. 环境与配置初始化
    accelerator, train_config = setup_environment(args,accelerator)

    # 2. 数据准备
    train_dataloader, eval_dataloader, test_dataloader = prepare_data(train_config)

    # 3. 模型与优化器准备
    model, optimizer, lr_scheduler = prepare_model_and_optimizer(train_config, train_dataloader)

    # 4. 使用accelerator.prepare包装所有组件
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # 根据eval_only参数决定是否进行训练
    logger.info("开始训练过程...")
    best_metric = run_training_loop(
        accelerator, train_config, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader,
        args.mi_loss_alpha, trial=trial
    )

    logger.info("训练完成")

    return best_metric

if __name__ == "__main__":
    main()
