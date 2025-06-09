# Ultralytics YOLO 🚀, AGPL-3.0 license
# Custom modules for YOLO-CXR based on https://ieeexplore.ieee.org/document/10720017

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, ConvTranspose, autopad # Assuming standard Conv and ConvTranspose are needed

__all__ = ["RefConv", "ECLA", "SFF"]

# ------------------- RefConv Implementation -------------------
# Based on description in YOLO-CXR paper (Section III-B) and general concept of reparameterization.
# NOTE: This is an interpretation using RepVGG style. For exact performance, refer to the original RefConv paper [48].
class RefConv(nn.Module):
    """
    Refocusing Convolution (RefConv) based on YOLO-CXR paper description (interpreted as RepVGG-style).
    Used to replace standard downsampling Conv layers in the backbone.
    Reference: https://ieeexplore.ieee.org/document/10720017 (Section III-B)
               Inspired by RepVGG: https://arxiv.org/abs/2101.03697
    """
    default_act = nn.SiLU()  # Default activation

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, deploy=False):
        super().__init__()
        assert k == 3, "RefConv interpretation uses k=3"
        assert s in [1, 2], "RefConv in YOLO-CXR replaces downsampling convs, stride should be 1 or 2."
        self.deploy = deploy
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Assume padding='same' if p is None
        padding = autopad(k, p, d)

        # Identity branch only exists if input and output channels are the same AND stride is 1
        self.bn = nn.BatchNorm2d(num_features=c1) if c1 == c2 and s == 1 else None

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, padding, dilation=d, groups=g, bias=True)
        else:
            # 3x3 branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )
            # 1x1 branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding=0, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )
            # TODO: Implement the actual Refocusing Transformation W_r based on Eq. (2) if details are available.
            # This RepVGG-style implementation serves as a placeholder for reparameterization.
            print(f"INFO: Using RepVGG-style placeholder for RefConv({c1}, {c2}, k={k}, s={s}).")

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        # Apply branches
        out_dense = self.rbr_dense(inputs)
        out_1x1 = self.rbr_1x1(inputs)
        out_id = self.bn(inputs) if self.bn is not None else 0

        # Sum branches
        return self.act(out_dense + out_1x1 + out_id)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias for the fused convolution."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_k_tensor(kernel1x1, k=3) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_k_tensor(self, kernel1x1, k):
        """Pads a 1x1 kernel to a kxk kernel."""
        if kernel1x1 is None:
            return 0
        pad = (k - 1) // 2
        return F.pad(kernel1x1, [pad, pad, pad, pad])

    def _fuse_bn_tensor(self, branch):
        """Fuses the batch normalization parameters into the convolutional kernel and bias."""
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        elif isinstance(branch, nn.BatchNorm2d): # Identity branch
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                # Create a 3x3 identity kernel for the identity branch
                kernel_value = torch.zeros((self.c1, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1 # Center pixel is 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
             return 0, 0 # Should not happen

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse parallel branches into a single RBR_REPARAM convolution for deployment."""
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            dilation=self.rbr_dense[0].dilation,
            groups=self.rbr_dense[0].groups,
            bias=True,
        ).to(self.rbr_dense[0].weight.device)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # Delete original branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

# ------------------- ECLA Module Implementation -------------------
# Based on YOLO-CXR paper (Section III-C, Figure 5)
class ECLA(nn.Module):
    """
    Efficient Channel and Local Attention (ECLA) module based on YOLO-CXR paper.
    Combines ECA-style channel attention with strip-pooling spatial attention.
    Reference: https://ieeexplore.ieee.org/document/10720017 (Section III-C, Figure 5)
               Inspired by ECA-Net (https://arxiv.org/abs/1910.03151) and Strip Pooling (https://arxiv.org/abs/2003.13328)
    """
    def __init__(self, c1, c2=None, k_size=None, gamma=2, b=1, sp_ksize=3, groups=32):
        super().__init__()
        # Sửa lỗi: Nếu c2 được cung cấp và khác c1, sử dụng lớp Conv 1x1 để điều chỉnh số kênh
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        self.need_channel_adjust = (c1 != c2)
        
        # Nếu cần thiết, thêm lớp conv 1x1 để điều chỉnh số kênh
        if self.need_channel_adjust:
            self.channel_adjust = Conv(c1, c2, k=1, s=1)
        
        # Use c2 as the working channel dimension for attention mechanisms
        c_working = c2

        # Channel Attention Branch (ECA-like)
        if k_size is None: # Auto calculate kernel size based on Eq. (4)
            t = int(abs((math.log2(c_working) + b) / gamma))
            k_size = max(t if t % 2 else t + 1, 3) # Make kernel size odd and minimum 3
        self.avg_pool_ca = nn.AdaptiveAvgPool2d(1)
        self.conv1d_ca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid_ca = nn.Sigmoid()

        # Spatial Attention Branch (Strip Pooling like)
        self.strip_pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Pool horizontally -> B, C, H, 1
        self.strip_pool_v = nn.AdaptiveAvgPool2d((1, None)) # Pool vertically -> B, C, 1, W

        # 1D Convs for local interaction along the strip
        sp_pad = (sp_ksize - 1) // 2
        # Process horizontal strip: needs Conv1d on H dimension (dim=2 after squeeze)
        self.conv1d_sp_h = nn.Conv1d(c_working, c_working, kernel_size=sp_ksize, padding=sp_pad, groups=c_working, bias=False)
        # Process vertical strip: needs Conv1d on W dimension (dim=2 after squeeze)
        self.conv1d_sp_v = nn.Conv1d(c_working, c_working, kernel_size=sp_ksize, padding=sp_pad, groups=c_working, bias=False)

        # GroupNorm as per paper ref [52]
        num_groups = max(1, c_working // groups if c_working >= groups else 1) # Ensure num_groups >= 1
        self.gn_h = nn.GroupNorm(num_groups=num_groups, num_channels=c_working)
        self.gn_v = nn.GroupNorm(num_groups=num_groups, num_channels=c_working)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        # Điều chỉnh số kênh nếu cần thiết
        if self.need_channel_adjust:
            x = self.channel_adjust(x)
            
        b, c, h, w = x.shape

        # --- Channel Attention (Eq. 5) ---
        y_ca = self.avg_pool_ca(x)  # B, C, 1, 1
        # Chỉnh sửa cách xử lý tensor để dễ hiểu hơn
        y_ca = y_ca.squeeze(-1).squeeze(-1)  # B, C
        y_ca = y_ca.unsqueeze(1)  # B, 1, C (Chuyển sang định dạng phù hợp cho Conv1d)
        y_ca = self.conv1d_ca(y_ca)  # B, 1, C
        y_ca = y_ca.unsqueeze(-1)  # B, 1, C, 1
        y_ca = y_ca.transpose(1, 2)  # B, C, 1, 1
        attn_ca = self.sigmoid_ca(y_ca)
        x_ca = x * attn_ca  # Channel-wise multiplication (Mc in paper)

        # --- Spatial Attention (Eq. 6, 7) ---
        # Horizontal strip pooling and processing
        y_sp_h = self.strip_pool_h(x_ca)  # B, C, H, 1
        y_sp_h = y_sp_h.squeeze(-1)  # B, C, H
        y_sp_h = self.conv1d_sp_h(y_sp_h)  # B, C, H
        y_sp_h = self.gn_h(y_sp_h)  # B, C, H
        attn_sp_h = self.sigmoid_sp(y_sp_h).unsqueeze(-1)  # B, C, H, 1 (S_H in paper)

        # Vertical strip pooling and processing
        y_sp_v = self.strip_pool_v(x_ca)  # B, C, 1, W
        y_sp_v = y_sp_v.squeeze(-2)  # B, C, W
        y_sp_v = self.conv1d_sp_v(y_sp_v)  # B, C, W
        y_sp_v = self.gn_v(y_sp_v)  # B, C, W
        attn_sp_v = self.sigmoid_sp(y_sp_v).unsqueeze(-2)  # B, C, 1, W (S_V in paper)

        # Apply spatial attention to the channel-attended features
        x_out = x_ca * attn_sp_h * attn_sp_v  # Element-wise multiplication

        return x_out

# ------------------- SFF Module Implementation -------------------
# Based on YOLO-CXR paper (Section III-D, Figure 6)
class ChannelAttentionSFF(nn.Module):
    """Channel Attention submodule for SFF, similar to SE block."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, channel // reduction) # Ensure hidden_dim is at least 1
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SFF(nn.Module):
    """
    Selective Feature Fusion (SFF) module based on YOLO-CXR paper.
    Fuses features from a shallow layer (low) and a deep layer (high).
    Reference: https://ieeexplore.ieee.org/document/10720017 (Section III-D, Figure 6)
    
    This module has been modified to work with the YAML configuration.
    It can operate in two modes:
    1. Single input mode (default): For use in YAML when feature maps are accessed via indexing
    2. Dual input mode: When explicitly called with two inputs (m_low, m_high)
    """
    def __init__(self, c1, c2=None, tconv_k=3):
        super().__init__()
        # We need to adapt this class to handle how it's used in YAML
        # In YAML, it's called with single layer like: [19, 1, SFF, [128]]
        # But conceptually it needs features from two layers
        
        # Store channels for later reference
        self.c1 = c1
        
        if c2 is None:
            c2 = c1
            
        self.c2 = c2
            
        # For single input mode - we'll access the other input from saved feature maps
        # Dựa vào cấu trúc YOLO-CXR với SFF, chúng ta giả định high-level features có số kênh gấp đôi low-level features
        c1_high = c2 * 2 
        
        self.tconv = ConvTranspose(c1_high, c2, k=tconv_k, s=2, p=autopad(tconv_k, p=None, d=1), bn=True, act=True)
        
        # Channel attention
        self.ca = ChannelAttentionSFF(c2)
        
        # Saved features map from network
        self.saved_features = None
        
    def forward(self, x):
        """
        Forward pass for SFF.
        
        Args:
            x: Có thể là:
               - Feature map đơn (sử dụng trong cấu hình YAML)
               - Tuple/list của hai feature maps (m_low, m_high) khi gọi trực tiếp trong code
               
        Note:
            Trong chế độ đầu vào đơn, chúng ta cần truy cập saved_features để lấy 
            feature map thứ hai (m_high). Điều này yêu cầu context bổ sung từ mạng.
        """
        # Determine input mode
        if isinstance(x, (list, tuple)) and len(x) == 2:
            # Dual input mode - explicit (m_low, m_high)
            m_low, m_high = x
        else:
            # Single input mode (as used in YAML)
            # Here x is the current layer (shallow features, M_low)
            m_low = x
            
            # Trong chế độ đầu vào đơn, chúng ta không thể truy cập feature map khác
            # nên tạo placeholder
            print(f"Warning: SFF used in single-input mode with input shape {m_low.shape}.")
            print("This requires future integration with the network's saved features.")
            
            # Tạo m_high placeholder có số kênh gấp đôi m_low và kích thước H,W bằng một nửa
            m_high = torch.zeros((m_low.shape[0], self.c2*2, m_low.shape[2]//2, m_low.shape[3]//2), 
                                device=m_low.device)
        
        # Upsample M_high (Eq. 8, part 1: TConv)
        m_b = self.tconv(m_high)

        # Ensure spatial dimensions match M_low (Eq. 8, part 2: BI - Bilinear Interpolation if needed)
        if m_b.shape[-2:] != m_low.shape[-2:]:
            m_b = F.interpolate(m_b, size=m_low.shape[-2:], mode='bilinear', align_corners=False)

        # Calculate channel attention weights from M_b
        attn_weights = self.ca(m_b) # CA(M_b)

        # Refine M_low using attention weights and fuse (Eq. 9)
        m_out = m_low * attn_weights + m_b # M_out = M_low * CA(M_b) + M_b

        return m_out

    def set_saved_features(self, features):
        """Set saved features from network for single-input mode."""
        self.saved_features = features
