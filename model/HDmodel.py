import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .stgcn_extractor import STGCN_Flexible
from .utils import get_normalize_adjacency_matrix
from typing import List, Tuple, Callable, Optional



def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    """
    计算指定维度上的softmax，支持多维度计算。

    Args:
        x (torch.Tensor): 输入张量。
        dim (int or tuple[int]): 计算softmax的维度。

    Returns:
        torch.Tensor: 应用softmax函数后的张量。
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class MutualInformationLoss(nn.Module):
    """
    计算互信息损失，用于鼓励路由的多样性。

    该损失对专家路由概率与聚合分布之间的低互信息进行惩罚。
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        """
        初始化MutualInformationLoss模块。

        Args:
            epsilon (float, optional): 添加到分母中的小值，以避免除零。默认为1e-4。
        """
        super(MutualInformationLoss, self).__init__()
        self.epsilon = epsilon  # 防止除零的小常数

    def check_nan(self, var: torch.Tensor, var_name: str) -> None:
        """
        检查给定张量是否包含NaN并打印警告。

        Args:
            var (torch.Tensor): 要检查的张量。
            var_name (str): 用于日志记录的变量名。
        """
        if torch.isnan(var).any():
            print(f"NaN detected in {var_name}:\n{var}")

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        基于路由张量phi计算互信息损失。

        Args:
            phi (torch.Tensor): 路由张量，形状为[batch_size, m, n, p]。

        Returns:
            torch.Tensor: 表示负互信息损失的标量张量。
        """
        batch_size, m, n, p = phi.shape
        phi = phi.reshape(phi.shape[0], phi.shape[1] * phi.shape[2] * phi.shape[3])
        phi = torch.softmax(phi, dim=1)
        phi = phi.reshape(phi.shape[0], m, n, p)
        
        # 计算边缘分布
        p_m = phi.sum(dim=(2, 3))
        p_t = phi.sum(dim=(1, 2))
        p_mt = phi.sum(dim=2)
        
        # 计算互信息公式的分母和分子
        denumerator = p_m.unsqueeze(2) * p_t.unsqueeze(1)
        numerator = p_mt
        log_term = torch.log(numerator / denumerator + 1e-10)
        mutual_info = torch.sum(p_mt * log_term, dim=(0, 1, 2))
      
        return -mutual_info


class ShuffleNetBackbone(nn.Module):
    """ShuffleNet-V2 X2.0 骨干网络，输出(B,T,2048)"""

    def __init__(self, output_dim = 128, pretrained: bool = True, freeze_backbone: bool = False, use_temporal_conv: bool = True):
        super().__init__()
        if pretrained:
            weights = models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
            self.net = models.shufflenet_v2_x2_0(weights=weights)
        else:
            self.net = models.shufflenet_v2_x2_0(weights=None)

        # 移除最后的分类层
        self.net.fc = nn.Identity()

        # 默认情况下，骨干网络是可训练的 (fine-tuning)
        if freeze_backbone:
            for param in self.net.parameters():
                param.requires_grad = False

        self.use_temporal_conv = use_temporal_conv
        if self.use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(2048, 2048, kernel_size=3, padding=1, groups=1),
                nn.LayerNorm(2048)
            )

        self.proj = nn.Linear(2048, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - 批次大小, 通道数, 时间步, 高度, 宽度
        Returns:
            feat: (B, T, 2048) - 时间序列特征
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        feat = self.net(x)

        feat = feat.view(B, T, -1)

        if self.use_temporal_conv:
            feat = self.temporal_conv[0](feat.permute(0, 2, 1))
            feat = feat.permute(0, 2, 1)  # (B, T, 2048)
            feat = self.temporal_conv[1](feat)

        feat = self.proj(feat)
        feat = self.norm(feat)
        return feat

    def get_optimizer_parameters(self, head_lr=1e-4):
        """
        返回用于差分学习率的参数组。
        - backbone: self.net
        - 头部: proj, norm, temporal_conv
        """
        base_lr = head_lr / 10 if not getattr(self, 'freeze_backbone', False) else 0.0
        params = []
        if not getattr(self, 'freeze_backbone', False):
            params.append({
                'params': list(self.net.parameters()),
                'lr': base_lr
            })
        head_params = []

        head_params.extend(self.proj.parameters())
        head_params.extend(self.norm.parameters())

        if self.use_temporal_conv and hasattr(self, 'temporal_conv'):
            head_params.extend(self.temporal_conv.parameters())

        if head_params:
            params.append({
                'params': head_params,
                'lr': head_lr
            })
            
        return params

class CrossViewFusion(nn.Module):

    def __init__(self, output_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.output_dim = output_dim
        self.num_heads = num_heads

        self.query_token = nn.Parameter(torch.randn(1, 1, output_dim))

        self.query_norm = nn.LayerNorm(output_dim)

        self.external_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )



    def forward(self, left: torch.Tensor, front: torch.Tensor,
                right: torch.Tensor, incar: torch.Tensor) -> torch.Tensor:

        B, T, _ = incar.shape
        external_kv = torch.cat([left, front, right,incar], dim=1)  # (B, 3T, 128)

        external_kv = self.query_norm(external_kv)
        query = self.query_token.expand(B, T, -1)  # (B, T, 128)
        query_norm = self.query_norm(query)

        external_feat, _ = self.external_attention(query_norm, external_kv, external_kv)
        external_feat = external_feat + query_norm
        fused_feat, _ = self.cross_attention(external_feat, incar, incar)
        fused_feat = fused_feat + external_feat
        return fused_feat

class MultiViewExtractor(nn.Module):
    """多视角特征提取器，处理四个视角（车内、前方、左侧、右侧）并融合"""

    def __init__(self, output_dim=128, pretrained=True, freeze_backbone=False, use_temporal_conv=True):
        super().__init__()

        self.incar_backbone = ShuffleNetBackbone(output_dim = output_dim, pretrained=pretrained, freeze_backbone=freeze_backbone, use_temporal_conv=use_temporal_conv)
        self.front_backbone = ShuffleNetBackbone(output_dim = output_dim, pretrained=pretrained, freeze_backbone=freeze_backbone, use_temporal_conv=use_temporal_conv)
        self.left_backbone = ShuffleNetBackbone(output_dim = output_dim, pretrained=pretrained, freeze_backbone=freeze_backbone, use_temporal_conv=use_temporal_conv)
        self.right_backbone = ShuffleNetBackbone(output_dim = output_dim, pretrained=pretrained, freeze_backbone=freeze_backbone, use_temporal_conv=use_temporal_conv)
        self.type_incar = nn.Parameter(torch.randn(1, 1, output_dim))
        self.type_front = nn.Parameter(torch.randn(1, 1, output_dim))
        self.type_left = nn.Parameter(torch.randn(1, 1, output_dim))
        self.type_right = nn.Parameter(torch.randn(1, 1, output_dim))

        self.pos_encoding = nn.Parameter(torch.randn(1, 16, output_dim))
        self.fusion_module = CrossViewFusion(output_dim=output_dim, num_heads=4)

    def forward(self, views_dict):

        # 提取每个视角的特征并添加类型编码和位置编码
        batch_size = views_dict["incar"].shape[0]

        incar_feat = self.incar_backbone(views_dict["incar"])
        incar_feat = incar_feat + self.type_incar.expand(batch_size, incar_feat.shape[1], -1)
        incar_feat = incar_feat + self.pos_encoding
        front_feat = self.front_backbone(views_dict["front"])
        front_feat = front_feat + self.type_front.expand(batch_size, front_feat.shape[1], -1)
        front_feat = front_feat + self.pos_encoding
        left_feat = self.left_backbone(views_dict["left"])
        left_feat = left_feat + self.type_left.expand(batch_size, left_feat.shape[1], -1)
        left_feat = left_feat + self.pos_encoding
        right_feat = self.right_backbone(views_dict["right"])
        right_feat = right_feat + self.type_right.expand(batch_size, right_feat.shape[1], -1)
        right_feat = right_feat + self.pos_encoding
        output = self.fusion_module(left_feat, front_feat, right_feat, incar_feat)  # (B, T, 128)

        return output

class STGCNExtractor(nn.Module):
    def __init__(self, num_nodes, in_channels=3, out_channels=128):
        super(STGCNExtractor, self).__init__()

        self.stgcn = STGCN_Flexible(num_nodes=num_nodes, num_features=in_channels, out_channels=out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, A_hat):
        x = x.squeeze(-1)  # (batch_size, in_channels, num_frames, num_nodes)
        out = self.stgcn(A_hat, x)  # (batch_size, num_nodes, out_channels)
        out = self.norm(out)
        return out

class HierarchicalAttentionFusionEncoder(nn.Module):
    
    def __init__(self, feature_dim=128, num_heads=4):
        super(HierarchicalAttentionFusionEncoder, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.type_face = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.type_body = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.type_posture = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.type_gesture = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.type_context = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_encoding_video = nn.Parameter(torch.randn(1, 16, feature_dim))
        self.pos_encoding_posture = nn.Parameter(torch.randn(1, 26, feature_dim))
        self.pos_encoding_gesture = nn.Parameter(torch.randn(1, 42, feature_dim))
        self.driver_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.driver_query_norm = nn.LayerNorm(feature_dim)
        self.driver_kv_norm = nn.LayerNorm(feature_dim)
        self.driver_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.driver_context_query_norm = nn.LayerNorm(feature_dim)

        self.driver_context_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.driver_context_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        self.context_driver_query_norm = nn.LayerNorm(feature_dim)

        self.context_driver_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.context_driver_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def forward(self, face_feat, body_feat, posture_feat, gesture_feat, context_feat):
        """
        Args:
            face_feat: 脸部特征 (B, T_face=16, feature_dim)
            body_feat: 身体特征 (B, T_body=16, feature_dim)
            posture_feat: 姿势特征 (B, T_posture=26, feature_dim)
            gesture_feat: 手势特征 (B, T_gesture=42, feature_dim)
            context_feat: 环境特征 (B, T_context=16, feature_dim)
        
        Returns:
            driver_repr: 驾驶员状态表征 (B, T, feature_dim)
            context_repr: 环境表征 (B, T, feature_dim)
        """
        batch_size = face_feat.shape[0]
        
        # 添加类型嵌入
        face_feat = face_feat + self.type_face.expand(batch_size, face_feat.shape[1], -1)
        body_feat = body_feat + self.type_body.expand(batch_size, body_feat.shape[1], -1)
        posture_feat = posture_feat + self.type_posture.expand(batch_size, posture_feat.shape[1], -1)
        gesture_feat = gesture_feat + self.type_gesture.expand(batch_size, gesture_feat.shape[1], -1)
        face_feat = face_feat + self.pos_encoding_video
        body_feat = body_feat + self.pos_encoding_video
        posture_feat = posture_feat + self.pos_encoding_posture
        gesture_feat = gesture_feat + self.pos_encoding_gesture

        driver_features = torch.cat([face_feat, body_feat, posture_feat, gesture_feat], dim=1)  # (B, T_face+T_body+T_posture+T_gesture, feature_dim)
        driver_features = self.driver_kv_norm(driver_features)
        query = self.driver_query.expand(batch_size, 8, -1)  # (B, 8, feature_dim)

        query_norm = self.driver_query_norm(query)
        driver_repr, _ = self.driver_attention(query_norm, driver_features, driver_features)
        driver_repr = driver_repr + query_norm

        context_query_norm = self.context_driver_query_norm(context_feat)
        driver_contextualized, _ = self.driver_context_attention(
            driver_repr, context_query_norm, context_query_norm
        )  # (B, 8, feature_dim)

        driver_ffn_output = self.driver_context_ffn(driver_repr)
        driver_contextualized = driver_contextualized + driver_ffn_output

        context_contextualized, _ = self.context_driver_attention(
            context_query_norm, driver_repr, driver_repr
        )  # (B, T_context, feature_dim)

        context_ffn_output = self.context_driver_ffn(context_query_norm)
        context_contextualized = context_contextualized + context_ffn_output

        return driver_contextualized, context_contextualized


class ExpertMLP(nn.Module):
    
    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        return self.norm(x + self.mlp(x))

class MTOEModule(nn.Module):
    
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        feature_dim: int,
        normalize: bool = True,
        expert_layer: Callable = ExpertMLP,
        **expert_kwargs
    ) -> None:

        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.slots_per_expert = num_tasks  # 每个专家的槽位数等于任务数
        self.normalize = normalize

        self.task_embeddings = nn.Embedding(num_tasks, feature_dim)

        self.phi = nn.Parameter(torch.zeros(feature_dim, num_experts, self.slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, mean=0, std=1 / feature_dim**0.5)
        
        self.criterion = MutualInformationLoss()
        self.experts = nn.ModuleList(
            [expert_layer(feature_dim=feature_dim, **expert_kwargs) for _ in range(num_experts)]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        MTOEModule的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为[batch_size, seq_len, feature_dim]。
                这是所有模态的token拼接而成的张量。

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]: 
                - 包含每个任务融合特征的列表，列表长度等于num_tasks
                - 组合权重张量，形状为[batch_size, seq_len, num_experts, slots_per_expert]
                - 互信息损失
        """
        assert x.shape[-1] == self.feature_dim, f"输入特征维度{x.shape[-1]}与层维度{self.feature_dim}不匹配"
        assert len(x.shape) == 3, f"输入应有3个维度，但有{len(x.shape)}"
        
        phi = self.phi

        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(phi, dim=0)

        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        mi_loss = self.criterion(logits)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        for task_idx in range(self.num_tasks):
            # 获取任务嵌入并扩展维度
            task_emb = self.task_embeddings(torch.tensor(task_idx, device=x.device))
            task_emb = task_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # 将任务嵌入添加到相应槽位
            xs[:, :, task_idx, :] = xs[:, :, task_idx, :] + task_emb

        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)],
            dim=1
        )
        y = torch.einsum("bnpd,bmnp->bmpd", ys, c)
        task_features = [y[:, :, task_idx, :] for task_idx in range(self.num_tasks)]
        
        return task_features, c, mi_loss

# 添加领域处理器
class DomainProcessor(nn.Module):

    def __init__(self, feature_dim=128, num_tasks=2, num_experts=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        
        # 使用MTOEModule处理多任务
        self.mtoe = MTOEModule(
            num_tasks=num_tasks,
            num_experts=num_experts,
            feature_dim=feature_dim,
            hidden_dim=feature_dim * 2
        )
        
    def forward(self, x):

        task_features, combine_weights, mi_loss = self.mtoe(x)
        pooled_features = [torch.mean(feat, dim=1) for feat in task_features]
        return pooled_features, combine_weights, mi_loss

# 定义主模型类
class HDMODEL(nn.Module):
    def __init__(self, mi_loss_alpha=0.1, soft_num_experts=8):
        super(HDMODEL, self).__init__()
        
        self.mi_loss_alpha = mi_loss_alpha

        # 获取并保存邻接矩阵（用于 ST-GCN 模块）
        posture_adjacency_matrix, gesture_adjacency_matrix = get_normalize_adjacency_matrix()

        self.posture_A_hat = posture_adjacency_matrix
        self.gesture_A_hat = gesture_adjacency_matrix

        self.face_extractor = ShuffleNetBackbone(output_dim=128, pretrained=True, freeze_backbone=False, use_temporal_conv=False)
        self.body_extractor = ShuffleNetBackbone(output_dim=128, pretrained=True, freeze_backbone=False, use_temporal_conv=False)
        self.context_extractor = MultiViewExtractor(output_dim=128, pretrained=True, freeze_backbone=False, use_temporal_conv=True)
        self.posture_extractor = STGCNExtractor(num_nodes=26, in_channels=3, out_channels=128)
        self.gesture_extractor = STGCNExtractor(num_nodes=42, in_channels=3, out_channels=128)
        self.fusion_encoder = HierarchicalAttentionFusionEncoder(feature_dim=128, num_heads=4)
        self.modalities = ["face", "body", "context", "posture", "gesture"]
        self.extractors = {
            "face": self.face_extractor,
            "body": self.body_extractor,
            "context": self.context_extractor,
            "posture": lambda x: self.posture_extractor(x[0], x[1]),
            "gesture": lambda x: self.gesture_extractor(x[0], x[1])
        }
        self.driver_processor = DomainProcessor(feature_dim=128, num_tasks=2, num_experts=soft_num_experts)
        self.environment_processor = DomainProcessor(feature_dim=128, num_tasks=2, num_experts=soft_num_experts)
        self.emotion_classifier = nn.Linear(128, 5)  # DER任务
        self.behavior_classifier = nn.Linear(128, 7)  # DBR任务
        self.context_classifier = nn.Linear(128, 3)  # TCR任务
        self.vehicle_classifier = nn.Linear(128, 5)  # VCR任务

    def forward(self, posture, gesture, context_views, body, face):
        """
        输入：
            posture: 姿态数据
            gesture: 手势数据
            context_views: 环境或上下文图像的多视角字典，包含incar, front, left, right四个视角
            body: 身体图像
            face: 脸部图像
        每个输入的具体格式需与对应提取器要求一致。
        """
        posture_A_hat = self.posture_A_hat.to(posture.device)
        gesture_A_hat = self.gesture_A_hat.to(gesture.device)
        raw_inputs = {
            "face": face,
            "body": body,
            "context": context_views,
            "posture": (posture, posture_A_hat),
            "gesture": (gesture, gesture_A_hat)
        }

        features = {}
        for mod in self.modalities:
            extractor = self.extractors.get(mod)
            input_data = raw_inputs[mod]
            feat = extractor(input_data)
            features[mod] = feat

        driver_contextualized, context_contextualized = self.fusion_encoder(
            features["face"], 
            features["body"], 
            features["posture"], 
            features["gesture"], 
            features["context"]
        )
        driver_task_features, _, driver_mi_loss = self.driver_processor(driver_contextualized)
        environment_task_features, _, environment_mi_loss = self.environment_processor(context_contextualized)
        
        mi_loss_terms = [m for m in (driver_mi_loss, environment_mi_loss) if m is not None]
        mi_loss = sum(mi_loss_terms) if mi_loss_terms else None
        emotion_feature, behavior_feature = driver_task_features
        context_feature, vehicle_feature = environment_task_features
        emotion_output = self.emotion_classifier(emotion_feature)
        behavior_output = self.behavior_classifier(behavior_feature)
        context_output = self.context_classifier(context_feature)
        vehicle_output = self.vehicle_classifier(vehicle_feature)
            
        outputs = (emotion_output, behavior_output, context_output, vehicle_output)

        if mi_loss is not None:
            return outputs, {"mi_loss": mi_loss}
        return outputs



if __name__ == '__main__':
    # 创建模型实例
    model = HDMODEL()
    
    # 测试用例
    print("\n测试模型前向传播...")
    
    # 创建随机输入数据
    batch_size = 1
    
    # 姿态数据 (B, C, T, N, 1) - 批次大小, 通道数, 时间步, 节点数, 1
    posture = torch.randn(batch_size, 3, 16, 26, 1)
    
    # 手势数据 (B, C, T, N, 1) - 批次大小, 通道数, 时间步, 节点数, 1
    gesture = torch.randn(batch_size, 3, 16, 42, 1)
    
    # 多视角上下文数据 - 字典包含四个视角
    context_views = {
        "incar": torch.randn(batch_size, 3, 16, 224, 224),   # 车内视角
        "front": torch.randn(batch_size, 3, 16, 224, 224),   # 前方视角
        "left": torch.randn(batch_size, 3, 16, 224, 224),    # 左侧视角
        "right": torch.randn(batch_size, 3, 16, 224, 224)    # 右侧视角
    }
    
    # 身体图像 (B, C, T, H, W) - 批次大小, 通道数, 时间步, 高度, 宽度
    body = torch.randn(batch_size, 3, 16, 224, 224)
    
    # 脸部图像 (B, C, T, H, W) - 批次大小, 通道数, 时间步, 高度, 宽度
    face = torch.randn(batch_size, 3, 16, 224, 224)
    
    # 将模型设置为评估模式
    model.eval()
    
    try:
        # 前向传播
        with torch.no_grad():
            emotion_output, behavior_output, context_output, vehicle_output = model(posture, gesture, context_views, body, face)
            
        # 打印输出形状
        print(f"情感输出形状: {emotion_output.shape}")  # 应为 [batch_size, 5]
        print(f"行为输出形状: {behavior_output.shape}")  # 应为 [batch_size, 7]
        print(f"上下文输出形状: {context_output.shape}")  # 应为 [batch_size, 3]
        print(f"车辆输出形状: {vehicle_output.shape}")  # 应为 [batch_size, 5]
        
        print("\n测试成功! 模型可以正常进行前向传播。")
    except Exception as e:
        print(f"\n测试失败! 错误信息: {str(e)}")
        
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 打印各个组件的参数量
    print("\n各组件参数量:")
    print(f"面部特征提取器: {sum(p.numel() for p in model.face_extractor.parameters()):,}")
    print(f"身体特征提取器: {sum(p.numel() for p in model.body_extractor.parameters()):,}")
    print(f"上下文特征提取器: {sum(p.numel() for p in model.context_extractor.parameters()):,}")
    print(f"姿态特征提取器: {sum(p.numel() for p in model.posture_extractor.parameters()):,}")
    print(f"手势特征提取器: {sum(p.numel() for p in model.gesture_extractor.parameters()):,}")
    print(f"融合编码器 fusion_encoder: {sum(p.numel() for p in model.fusion_encoder.parameters()):,}")
    print(f"驾驶员领域处理器 driver_processor: {sum(p.numel() for p in model.driver_processor.parameters()):,}")
    print(f"环境领域处理器 environment_processor: {sum(p.numel() for p in model.environment_processor.parameters()):,}")
    print(f"分类器: {sum(p.numel() for p in model.emotion_classifier.parameters()) + sum(p.numel() for p in model.behavior_classifier.parameters()) + sum(p.numel() for p in model.context_classifier.parameters()) + sum(p.numel() for p in model.vehicle_classifier.parameters()):,}")

    # ========== 精确手工计算三个模块 FLOPs（batch_size=1, soft_num_experts=8） ==========
    def count_linear_flops(in_features: int, out_features: int, seq_len: int) -> int:
        # 乘加各算2 FLOPs
        return 2 * in_features * out_features * seq_len

    def count_mha_flops(L_q: int, L_k: int, d_model: int, num_heads: int) -> int:
        d_k = d_model // num_heads
        # Q/K/V projection
        flops_qkv = 3 * count_linear_flops(d_model, d_model, L_q if L_q == L_k else (L_q + L_k))
        # 注意力得分与加权和
        flops_attn = 2 * L_q * L_k * d_k * num_heads * 2  # QK^T 和 attn@V
        # 输出投影
        flops_out = count_linear_flops(d_model, d_model, L_q)
        return flops_qkv + flops_attn + flops_out

    # ---------- fusion_encoder FLOPs ----------
    B = 1
    d_model = 128
    num_heads = 4
    T_face = 16
    T_body = 16
    T_posture = 26
    T_gesture = 42
    T_context = 16
    T_driver_tokens = 8

    # 类型嵌入 + 位置编码（逐元素加法，1 FLOP/元素）
    elems_face = B * T_face * d_model
    elems_body = B * T_body * d_model
    elems_posture = B * T_posture * d_model
    elems_gesture = B * T_gesture * d_model
    elems_context = B * T_context * d_model
    flops_add_embeddings = (
        2 * elems_face +            # face: type + pos
        2 * elems_body +            # body: type + pos
        elems_posture * 2 +         # posture: type + pos
        elems_gesture * 2 +         # gesture: type + pos
        0                           # context 已在 MultiViewExtractor 中加过
    )

    # driver_features 拼接仅是 view，无 FLOPs
    L_driver_kv = T_face + T_body + T_posture + T_gesture  # 100

    # driver_query expand 视为 0 FLOPs；LayerNorm 约 5 FLOPs/元素
    elems_driver_query = B * T_driver_tokens * d_model
    flops_driver_query_ln = 5 * elems_driver_query

    # driver_kv_norm LayerNorm
    elems_driver_kv = B * L_driver_kv * d_model
    flops_driver_kv_ln = 5 * elems_driver_kv

    # 1) driver_attention
    flops_driver_attn = count_mha_flops(
        L_q=T_driver_tokens,
        L_k=L_driver_kv,
        d_model=d_model,
        num_heads=num_heads,
    )

    # 残差 driver_repr + query_norm
    flops_driver_residual = elems_driver_query

    # 2) driver_context_attention（driver 作为 Q，context 作为 KV）
    elems_context_ln = B * T_context * d_model
    flops_context_driver_query_ln = 5 * elems_context_ln
    flops_driver_context_attn = count_mha_flops(
        L_q=T_driver_tokens,
        L_k=T_context,
        d_model=d_model,
        num_heads=num_heads,
    )

    # driver_context_ffn：Linear(128->256->128) + ReLU（忽略）
    flops_driver_ffn = (
        count_linear_flops(d_model, d_model * 2, T_driver_tokens) +
        count_linear_flops(d_model * 2, d_model, T_driver_tokens)
    )
    # 残差 driver_contextualized + driver_ffn_output
    flops_driver_context_residual = elems_driver_query

    # 3) context_driver_attention（context 作为 Q，driver 作为 KV）
    flops_context_driver_attn = count_mha_flops(
        L_q=T_context,
        L_k=T_driver_tokens,
        d_model=d_model,
        num_heads=num_heads,
    )

    # context_driver_ffn：对 context_feat 上的每个 token
    flops_context_ffn = (
        count_linear_flops(d_model, d_model * 2, T_context) +
        count_linear_flops(d_model * 2, d_model, T_context)
    )
    # 残差 context_contextualized + context_ffn_output
    flops_context_residual = elems_context

    flops_fusion_total = (
        flops_add_embeddings +
        flops_driver_query_ln +
        flops_driver_kv_ln +
        flops_driver_attn +
        flops_driver_residual +
        flops_context_driver_query_ln +
        flops_driver_context_attn +
        flops_driver_ffn +
        flops_driver_context_residual +
        flops_context_driver_attn +
        flops_context_ffn +
        flops_context_residual
    )

    print(f"\n[Hand] fusion_encoder FLOPs: {flops_fusion_total/1e6:.3f} MFLOPs ({flops_fusion_total/1e9:.6f} GFLOPs)")

    # ---------- DomainProcessor / MTOEModule FLOPs ----------
    def count_mtoe_flops(seq_len: int, feature_dim: int, num_experts: int, num_tasks: int) -> int:
        B = 1
        D = feature_dim
        M = seq_len
        N = num_experts
        P = num_tasks

        # einsum("bmd,dnp->bmnp")
        flops_logits = 2 * B * M * D * N * P
        # d = softmax(logits, dim=1) 和 c = softmax(logits, dim=(2,3)) 近似为 5 FLOPs/元素
        elems_logits = B * M * N * P
        flops_softmax = 5 * elems_logits * 2
        # xs = einsum("bmd,bmnp->bnpd")
        flops_xs = 2 * B * M * N * P * D
        # 为每个任务添加 task embedding：加法
        flops_task_emb = B * N * P * D
        # experts：每个 expert 对所有 (N*P) 槽位做 MLP
        # 先把 experts 输入视为 (B * N * P, D)
        slots = B * N * P
        flops_experts = slots * (
            count_linear_flops(D, D * 2, 1) +
            count_linear_flops(D * 2, D, 1)
        )
        # y = einsum("bnpd,bmnp->bmpd")
        flops_y = 2 * B * M * N * P * D
        # 时序池化 mean：加法 + 除法 ~ 2 FLOPs/元素
        elems_pool = B * P * D
        flops_pool = 2 * elems_pool
        return flops_logits + flops_softmax + flops_xs + flops_task_emb + flops_experts + flops_y + flops_pool

    soft_num_experts = 8
    num_tasks = 2
    # driver_processor 输入 seq_len=8
    flops_driver_proc = count_mtoe_flops(seq_len=T_driver_tokens, feature_dim=d_model, num_experts=soft_num_experts, num_tasks=num_tasks)
    # environment_processor 输入 seq_len=16
    flops_env_proc = count_mtoe_flops(seq_len=T_context, feature_dim=d_model, num_experts=soft_num_experts, num_tasks=num_tasks)

    print(f"[Hand] driver_processor FLOPs: {flops_driver_proc/1e6:.3f} MFLOPs ({flops_driver_proc/1e9:.6f} GFLOPs)")
    print(f"[Hand] environment_processor FLOPs: {flops_env_proc/1e6:.3f} MFLOPs ({flops_env_proc/1e9:.6f} GFLOPs)")

    # ========== 单独统计 fusion_encoder 和 DomainProcessor FLOPs ==========
    try:
        from thop import profile
        # fusion_encoder FLOPs（只返回 driver_contextualized，避免 tuple 导致 FLOPs 为 0）
        class FusionEncoderOnly(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.fusion_encoder = model.fusion_encoder
            def forward(self, face, body, posture, gesture, context):
                driver_contextualized, _ = self.fusion_encoder(face, body, posture, gesture, context)
                return driver_contextualized
        fusion_encoder_model = FusionEncoderOnly(model)
        fusion_encoder_model.eval()
        # 生成特征输入
        with torch.no_grad():
            face_feat = model.face_extractor(face)
            body_feat = model.body_extractor(body)
            context_feat = model.context_extractor(context_views)
            posture_A_hat = model.posture_A_hat.to(posture.device)
            gesture_A_hat = model.gesture_A_hat.to(gesture.device)
            posture_feat = model.posture_extractor(posture, posture_A_hat)
            gesture_feat = model.gesture_extractor(gesture, gesture_A_hat)
        fusion_inputs = (face_feat, body_feat, posture_feat, gesture_feat, context_feat)
        flops_fusion, params_fusion = profile(fusion_encoder_model, inputs=fusion_inputs, verbose=False)
        print(f"\n[thop] fusion_encoder 参数量: {params_fusion:,}")
        print(f"[thop] fusion_encoder FLOPs raw: {flops_fusion}")
        print(f"[thop] fusion_encoder FLOPs: {flops_fusion/1e6:.3f} MFLOPs ({flops_fusion/1e9:.6f} GFLOPs)")

        # driver_processor FLOPs（只返回第一个 pooled_features，避免 tuple 导致 FLOPs 为 0）
        class DriverProcessorOnly(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.driver_processor = model.driver_processor
            def forward(self, x):
                pooled_features, _, _ = self.driver_processor(x)
                return pooled_features[0]
        driver_processor_model = DriverProcessorOnly(model)
        driver_processor_model.eval()
        # 用融合后的driver特征作为输入
        with torch.no_grad():
            driver_contextualized, _ = model.fusion_encoder(face_feat, body_feat, posture_feat, gesture_feat, context_feat)
        flops_driver, params_driver = profile(driver_processor_model, inputs=(driver_contextualized,), verbose=False)
        print(f"[thop] driver_processor 参数量: {params_driver:,}")
        print(f"[thop] driver_processor FLOPs raw: {flops_driver}")
        print(f"[thop] driver_processor FLOPs: {flops_driver/1e6:.3f} MFLOPs ({flops_driver/1e9:.6f} GFLOPs)")

        # environment_processor FLOPs（只返回第一个 pooled_features，避免 tuple 导致 FLOPs 为 0）
        class EnvProcessorOnly(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.environment_processor = model.environment_processor
            def forward(self, x):
                pooled_features, _, _ = self.environment_processor(x)
                return pooled_features[0]
        env_processor_model = EnvProcessorOnly(model)
        env_processor_model.eval()
        # 用融合后的environment特征作为输入
        with torch.no_grad():
            _, environment_contextualized = model.fusion_encoder(face_feat, body_feat, posture_feat, gesture_feat, context_feat)
        flops_env, params_env = profile(env_processor_model, inputs=(environment_contextualized,), verbose=False)
        print(f"[thop] environment_processor 参数量: {params_env:,}")
        print(f"[thop] environment_processor FLOPs raw: {flops_env}")
        print(f"[thop] environment_processor FLOPs: {flops_env/1e6:.3f} MFLOPs ({flops_env/1e9:.6f} GFLOPs)")
    except ImportError:
        print("未安装thop库，无法统计FLOPs。可通过 pip install thop 安装。")
    except Exception as e:
        print(f"单独FLOPs统计失败: {e}")
