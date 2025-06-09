# Tệp: modules/ans_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv
import math

class ANSModule(nn.Module):
    """
    Adaptive Noise Suppression Module (ANS) cho NSEC-YOLO.
    Giảm nhiễu nền và cải thiện biểu diễn đặc trưng thông qua cơ chế chú ý kênh và không gian.
    """
    def __init__(self, channels):
        super(ANSModule, self).__init__()
        self.channels = channels
        
        # Convolution với các kernel size khác nhau
        self.conv3x3 = Conv(channels, channels, 3, 1, autopad=True)
        self.conv5x5 = Conv(channels, channels, 5, 1, autopad=True)
        self.conv7x7 = Conv(channels, channels, 7, 1, autopad=True)
        
        # Channel attention module
        # Tính toán kernel size k dựa trên số channels theo công thức trong bài báo
        k = max(3, int(abs(math.log2(channels) / 2 + 1)))
        k = k if k % 2 == 1 else k + 1  # Đảm bảo k là số lẻ
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial attention module
        self.conv1d_h = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=1)
        self.conv1d_v = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=1)
        
        # Batch normalization cho channel attention
        self.bn = nn.BatchNorm2d(channels)
        
        # Spatial batch normalization (pixel normalization)
        self.bn_spatial = nn.BatchNorm2d(channels)
        
        # Tham số hình phạt
        self.penalty_coeff = 0.01  # Tham số p trong Eq.(6)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            Conv(channels, channels, 3, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Forward pass của module ANS.
        
        Args:
            x: Tensor đầu vào [batch_size, channels, height, width]
            
        Returns:
            Tensor đầu ra đã được lọc nhiễu
        """
        batch_size, c, h, w = x.size()
        
        # Trích xuất đặc trưng với các kernel khác nhau
        M1 = self.conv3x3(x)  # Eq.(7)
        M2 = self.conv5x5(x)
        M3 = self.conv7x7(x)
        
        # Channel attention trên M2
        y = self.avg_pool(M2)  # [batch_size, channels, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [batch_size, 1, channels]
        y = self.conv1d(y)  # [batch_size, 1, channels]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [batch_size, channels, 1, 1]
        
        # Lấy scale factor từ batch normalization (Eq. 4)
        M2_bn = self.bn(M2)
        gamma = self.bn.weight  # [channels]
        # Normalize weight để tạo thành trọng số kênh (Eq. 2)
        W_gamma = gamma / (gamma.sum() + 1e-8)
        W_gamma = W_gamma.view(1, -1, 1, 1)
        
        # Channel attention mask (Eq. 4)
        Mc = self.sigmoid_channel(W_gamma * M2_bn)
        
        # Spatial attention trên M3
        # Horizontal strip-pooling
        sp_h = F.adaptive_avg_pool2d(M3, (h, 1))  # [batch_size, channels, h, 1]
        sp_h = sp_h.squeeze(-1)  # [batch_size, channels, h]
        sp_h = self.conv1d_h(sp_h)  # [batch_size, channels, h]
        sp_h = sp_h.unsqueeze(-1)  # [batch_size, channels, h, 1]
        
        # Vertical strip-pooling
        sp_v = F.adaptive_avg_pool2d(M3, (1, w))  # [batch_size, channels, 1, w]
        sp_v = sp_v.squeeze(-2)  # [batch_size, channels, w]
        sp_v = self.conv1d_v(sp_v)  # [batch_size, channels, w]
        sp_v = sp_v.unsqueeze(-2)  # [batch_size, channels, 1, w]
        
        # Spatial batch normalization
        lambda_params = self.bn_spatial.weight
        W_lambda = lambda_params / (lambda_params.sum() + 1e-8)
        W_lambda = W_lambda.view(1, -1, 1, 1)
        
        # Spatial attention masks (Eq. 5)
        Ms_h = self.sigmoid_channel(W_lambda * self.bn_spatial(sp_h))
        Ms_v = self.sigmoid_channel(W_lambda * self.bn_spatial(sp_v))
        
        # Kết hợp các masks với feature M1 (Eq. 7)
        combined = M1 * Mc * Ms_h * Ms_v
        
        # Fusion (Eq. 7)
        M_out = self.fusion(combined)
        
        return M_out