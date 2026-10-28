import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TaskSpecificAdapter(nn.Module):

    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        
        self.valence_adapter = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, in_dim // 2)
        )
        
        self.arousal_adapter = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, in_dim // 2)
        )
        
        self.valence_classifier = nn.Linear(in_dim // 2, num_classes)
        self.arousal_classifier = nn.Linear(in_dim // 2, num_classes)
        
    def forward(self, x):
        valence_features = self.valence_adapter(x)
        arousal_features = self.arousal_adapter(x)
        
        class_outputs = {
            'valence': self.valence_classifier(valence_features),
            'arousal': self.arousal_classifier(arousal_features)
        }
        
        return class_outputs

class IPPG1DCNN(nn.Module):
    # 一维卷积神经网络，用于从 IPPG 信号中提取生理特征
    def __init__(self, out_dim=248, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, out_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        return x

class ImprovedMultiModalMultiTaskClassifier(nn.Module):
    # 基于 ResNet-50、AMIL(60选16) 与跨注意力机制的多模态多任务情感分类模型
    def __init__(
        self,
        emotion_dims=['valence', 'arousal'],
        num_classes=2,
        hrv_size=8,
        fusion_dim=768,
        num_heads=4,
        dim_feedforward=1024,
        num_encoder_layers=3,
        transformer_dropout_rate=0.2,
        cls_dropout_rate=0.3,
        fusion_type='improved_cross_attention',
        instance_selection_method='amil_topk',
        num_select=16,
        num_instances=60,
        image_dim=1024,
    ):
        super().__init__()
        self.emotion_dims = emotion_dims
        self.fusion_type = fusion_type
        self.instance_selection_method = instance_selection_method
        self.num_select = num_select
        self.num_instances = num_instances
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.img_proj = nn.Linear(2048, image_dim)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        for name, param in self.resnet.named_parameters():
            if name.startswith('layer4') or 'bn' in name:
                param.requires_grad = True

        self.hrv_proj = nn.Sequential(
            nn.LayerNorm(hrv_size),
            nn.Linear(hrv_size, hrv_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hrv_size * 2, hrv_size)
        )
        
        self.ippg_cnn = IPPG1DCNN(out_dim=248)
        self.ippg_proj = nn.Linear(248, 248)
        
        self.ippg_hrv_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hrv_size)
        self.img_dropout = nn.Dropout(0.3)
        self.hrv_dropout = nn.Dropout(0.2)
        self.ippg_dropout = nn.Dropout(0.2)
        
        self.improved_fusion = ImprovedMultiModalFusion(
            img_dim=image_dim,
            physio_dim=256,
            fusion_dim=fusion_dim
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.dropout = nn.Dropout(cls_dropout_rate)
        
        self.task_adapter = TaskSpecificAdapter(fusion_dim, num_classes)
        
        self.attention_V = nn.Linear(image_dim, image_dim)
        self.attention_w = nn.Linear(image_dim, 1)
        self.temperature = 0.5
        
        self.register_buffer('image_importance', torch.tensor(0.0))
        self.register_buffer('physio_importance', torch.tensor(0.0))
        
        nn.init.normal_(self.cls_token, std=0.02)

    def select_instances(self, images_embedding):
        # 基于 Top-K 注意力机制的实例选择方法
        batch_size, num_instances = images_embedding.shape[:2]
        select_num = min(self.num_select, num_instances)
        
        if self.instance_selection_method == 'amil_topk':
            hidden = torch.tanh(self.attention_V(images_embedding))
            weights = self.attention_w(hidden).squeeze(-1)
            weights = weights / self.temperature
            weights = F.softmax(weights, dim=1)
            
            _, indices = torch.topk(weights, select_num, dim=1)
            selected_embeddings = []
            for i in range(batch_size):
                selected = images_embedding[i, indices[i], ...]
                selected_embeddings.append(selected)
            return torch.stack(selected_embeddings)

    def forward(self, hrv_data, images_data, ippg_signal):
        # 模型前向传播计算
        batch_size, num_instances = images_data.shape[:2]

        images_data = images_data.contiguous().view(-1, 3, 224, 224)
        img_feats = self.resnet(images_data)
        img_feats = self.img_proj(img_feats)
        img_feats = self.img_dropout(img_feats)
        img_feats = img_feats.view(batch_size, num_instances, -1)
        selected_embeddings = self.select_instances(img_feats)

        if hrv_data.dim() == 3:
            hrv_data = hrv_data.mean(dim=1)
        elif hrv_data.dim() != 2:
            raise ValueError("HRV data must be 2D or 3D tensor")
        
        hrv_data = self.layernorm(hrv_data)
        hrv_feat = self.hrv_proj(hrv_data)
        hrv_feat = self.activation(hrv_feat)
        hrv_feat = self.hrv_dropout(hrv_feat)

        if ippg_signal.dim() == 4:
            ippg_signal = ippg_signal.squeeze(2)
        ippg_feat = self.ippg_cnn(ippg_signal)
        ippg_feat = self.ippg_proj(ippg_feat)
        ippg_feat = self.activation(ippg_feat)
        ippg_feat = self.ippg_dropout(ippg_feat)

        ippg_hrv_feat = torch.cat([hrv_feat, ippg_feat], dim=1)
        ippg_hrv_embedding = self.ippg_hrv_proj(ippg_hrv_feat).unsqueeze(1)

        if self.fusion_type == 'improved_cross_attention':
            fused_features = self.improved_fusion(
                selected_embeddings,
                ippg_hrv_embedding
            )
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            multi_embedding = torch.cat([cls_token, fused_features.unsqueeze(1)], dim=1)
            
        multi_embedding = self.transformer_encoder(multi_embedding)
        cls_output = multi_embedding[:, 0, :]
        cls_output = self.dropout(cls_output)

        class_outputs = self.task_adapter(cls_output)
        
        with torch.no_grad():
            if hasattr(self, 'improved_fusion'):
                weights = F.softmax(self.improved_fusion.modality_weights, dim=0)
                self.image_importance = 0.9 * self.image_importance + 0.1 * weights[0]
                self.physio_importance = 0.9 * self.physio_importance + 0.1 * weights[1]
        
        return class_outputs

    def print_gates(self):
        # 打印并监控各模态权重与重要性
        print(f"  image_importance: {self.image_importance.item():.4f}")
        print(f"  physio_importance: {self.physio_importance.item():.4f}")
        if hasattr(self.improved_fusion, 'modality_weights'):
            weights = F.softmax(self.improved_fusion.modality_weights, dim=0)
            print(f"  learned_weights - img: {weights[0].item():.4f}, physio: {weights[1].item():.4f}")
