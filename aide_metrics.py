import numpy as np
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 情绪标签的粗粒度映射
EMOTION_COARSE_MAP = {
    0: 'Negative',  # Anxiety
    1: 'Neutral',   # Peace
    2: 'Negative',  # Weariness
    3: 'Positive',  # Happiness
    4: 'Negative',  # Anger
}

# 驾驶行为标签的粗粒度映射
DRIVER_BEHAVIOR_COARSE_MAP = {
    0: 'Abnormal',  # Smoking
    1: 'Abnormal',  # Making Phone
    2: 'Abnormal',  # Looking Around
    3: 'Abnormal',  # Dozing Off
    4: 'Normal',    # Normal Driving
    5: 'Abnormal',  # Talking
    6: 'Abnormal',  # Body Movement
}

# 定义标签列表，用于混淆矩阵绘制
EMOTION_LABELS = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABELS = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking', 'Body Movement']
SCENE_CENTRIC_CONTEXT_LABELS = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABELS = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']
EMOTION_COARSE_LABELS = ['Negative', 'Neutral', 'Positive']
DRIVER_BEHAVIOR_COARSE_LABELS = ['Normal', 'Abnormal']

def plot_confusion_matrix(y_true, y_pred, class_labels, task_name="", dpi=300):
    """
    创建并返回一个标准化的混淆矩阵图
    
    Args:
        y_true: 真实标签的列表或数组
        y_pred: 预测标签的列表或数组
        class_labels: 类别名称的列表
        task_name: 任务名称，用于图表标题
        dpi: 图像的分辨率（每英寸点数）
        
    Returns:
        matplotlib的figure对象
    """
    # 计算归一化的混淆矩阵（按真实标签/行归一化）
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # 创建figure和axes对象
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    
    # 使用seaborn绘制热力图
    sns.heatmap(
        cm, 
        annot=True,         # 在格子上显示数值
        cmap='Blues',       # 使用蓝色系配色
        fmt='.2f',          # 将数值格式化为带两位小数的浮点数
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        annot_kws={'size': 18}  # 增大数值标注的字体大小
    )
    
    # Set chart title and axis labels
    ax.set_title(task_name, fontsize=18)
    ax.set_xlabel('Predicted Labels', fontsize=16)
    ax.set_ylabel('True Labels', fontsize=16)
    
    # 设置x轴和y轴标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 设置x轴标签倾斜
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局，确保标签完整显示
    plt.tight_layout()
    
    return fig

# 初始化评估器
accuracy_metric = evaluate.load(path= "./metrics/accuracy")
f1_metric = evaluate.load(path= "./metrics/f1")
def compute_metrics(eval_pred):
    """
    计算评估指标并生成混淆矩阵。

    Args:
        eval_pred: 包含预测值和真实标签的元组。

    Returns:
        字典，包含所有需要的评估指标和混淆矩阵图。
    """
    logits, labels = eval_pred
    # 假设模型输出是一个包含四个元素的元组

    emotion_logits, behavior_logits, context_logits, vehicle_logits = logits
    emotion_labels, behavior_labels, context_labels, vehicle_labels = labels

    # 获取预测值
    # 返回样本在每个类别中的最大值，然后展平为一维数组（Numpy数组）
    emotion_preds = np.argmax(emotion_logits, axis=-1).flatten()
    behavior_preds = np.argmax(behavior_logits, axis=-1).flatten()
    context_preds = np.argmax(context_logits, axis=-1).flatten()
    vehicle_preds = np.argmax(vehicle_logits, axis=-1).flatten()

    # 真实标签
    emotion_true = emotion_labels.flatten()
    behavior_true = behavior_labels.flatten()
    context_true = context_labels.flatten()
    vehicle_true = vehicle_labels.flatten()

    # 生成混淆矩阵
    emotion_cm_fig = plot_confusion_matrix(
        emotion_true,
        emotion_preds,
        EMOTION_LABELS,
        "Emotion"
    )
    
    behavior_cm_fig = plot_confusion_matrix(
        behavior_true,
        behavior_preds,
        DRIVER_BEHAVIOR_LABELS,
        "Behavior"
    )
    
    context_cm_fig = plot_confusion_matrix(
        context_true,
        context_preds,
        SCENE_CENTRIC_CONTEXT_LABELS,
        "Context"
    )
    
    vehicle_cm_fig = plot_confusion_matrix(
        vehicle_true,
        vehicle_preds,
        VEHICLE_BASED_CONTEXT_LABELS,
        "Vehicle"
    )

    # 初始化结果字典
    results = {}

    # 定义任务名称和对应的预测值、真实值
    tasks = {
        'emotion': (emotion_preds, emotion_true),
        'behavior': (behavior_preds, behavior_true),
        'context': (context_preds, context_true),
        'vehicle': (vehicle_preds, vehicle_true)
    }

    All_F1 = []
    All_Acc = []
    # 计算每个任务的 Acc 和 F1
    for task_name, (preds, trues) in tasks.items():
        acc = accuracy_metric.compute(predictions=preds, references=trues)['accuracy']
        f1 = f1_metric.compute(predictions=preds, references=trues, average='macro')['f1']
        results[f'{task_name}_acc'] = acc
        results[f'{task_name}_f1'] = f1
        All_Acc.append(acc)
        All_F1.append(f1)

    # 计算粗粒度的指标（针对 emotion 和 behavior）
    # Emotion 粗粒度标签
    emotion_coarse_preds = np.array([EMOTION_COARSE_MAP[pred.item()] for pred in emotion_preds])
    emotion_coarse_true = np.array([EMOTION_COARSE_MAP[true.item()] for true in emotion_true])

    # 将情绪粗粒度标签映射到数值
    emotion_coarse_label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    emotion_coarse_preds_num = np.array([emotion_coarse_label_map[label] for label in emotion_coarse_preds])
    emotion_coarse_true_num = np.array([emotion_coarse_label_map[label] for label in emotion_coarse_true])

    # 计算情绪的 CG-Acc 和 CG-F1
    emotion_cg_acc = accuracy_metric.compute(predictions=emotion_coarse_preds_num, references=emotion_coarse_true_num)['accuracy']
    emotion_cg_f1 = f1_metric.compute(predictions=emotion_coarse_preds_num, references=emotion_coarse_true_num, average='macro')['f1']
    results['emotion_cg_acc'] = emotion_cg_acc
    results['emotion_cg_f1'] = emotion_cg_f1

    # 生成粗粒度情绪混淆矩阵
    emotion_coarse_cm_fig = plot_confusion_matrix(
        emotion_coarse_true_num,
        emotion_coarse_preds_num,
        EMOTION_COARSE_LABELS,
        "Coarse Emotion"
    )

    # Behavior 粗粒度标签
    behavior_coarse_preds = np.array([DRIVER_BEHAVIOR_COARSE_MAP[pred.item()] for pred in behavior_preds])
    behavior_coarse_true = np.array([DRIVER_BEHAVIOR_COARSE_MAP[true.item()] for true in behavior_true])

    # 将驾驶行为粗粒度标签映射到数值
    behavior_coarse_label_map = {'Normal': 0, 'Abnormal': 1}
    behavior_coarse_preds_num = np.array([behavior_coarse_label_map[label] for label in behavior_coarse_preds])
    behavior_coarse_true_num = np.array([behavior_coarse_label_map[label] for label in behavior_coarse_true])

    # 计算驾驶行为的 CG-Acc 和 CG-F1
    behavior_cg_acc = accuracy_metric.compute(predictions=behavior_coarse_preds_num, references=behavior_coarse_true_num)['accuracy']
    behavior_cg_f1 = f1_metric.compute(predictions=behavior_coarse_preds_num, references=behavior_coarse_true_num, average='macro')['f1']
    results['behavior_cg_acc'] = behavior_cg_acc
    results['behavior_cg_f1'] = behavior_cg_f1

    # 生成粗粒度驾驶行为混淆矩阵
    behavior_coarse_cm_fig = plot_confusion_matrix(
        behavior_coarse_true_num,
        behavior_coarse_preds_num,
        DRIVER_BEHAVIOR_COARSE_LABELS,
        "Coarse Behavior"
    )

    # 计算所有任务的平均 F1
    All_F1.append(emotion_cg_f1)
    All_F1.append(behavior_cg_f1)
    results['All_F1'] = np.mean(All_F1)
    
    # 计算四个主要任务的平均Acc
    results['mAcc'] = np.mean(All_Acc)

    # 计算所有任务的平均Acc
    All_Acc.append(emotion_cg_acc)
    All_Acc.append(behavior_cg_acc)
    results['All_Acc'] = np.mean(All_Acc)

    # 添加混淆矩阵到结果中
    results['confusion_matrices'] = (emotion_cm_fig, behavior_cm_fig, context_cm_fig, vehicle_cm_fig, emotion_coarse_cm_fig, behavior_coarse_cm_fig)

    return results