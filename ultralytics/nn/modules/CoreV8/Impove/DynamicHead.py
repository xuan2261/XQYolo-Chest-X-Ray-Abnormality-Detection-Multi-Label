# # Copyright (c) OpenMMLab. All rights reserved.
# # Modified to remove mmcv dependencies and use torchvision.ops.DeformConv2d
# # Includes fixes for offset/mask shape and spatial dimension mismatches.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# try:
#     # Attempt to import DeformConv2d from torchvision
#     from torchvision.ops import DeformConv2d
# except ImportError:
#     # Provide a helpful error message if torchvision or the op is missing
#     print("ERROR: torchvision.ops.DeformConv2d not found.")
#     print("Please install or update torchvision: pip install -U torchvision")
#     # Re-raise the error to stop execution if the dependency is critical
#     raise
# import math

# # --- Helper Functions (Standard PyTorch/common practice) ---

# def _ntuple(n):
#     """Create a tuple generator."""
#     def parse(x):
#         if isinstance(x, (list, tuple)):
#             return x
#         return tuple([x] * n)
#     return parse

# _pair = _ntuple(2) # Often used for kernel_size, stride, padding

# # --- Initialization Functions (Replacing mmcv.cnn.init) ---

# def constant_init(module, val, bias=0):
#     """Initialize module parameters with constant values."""
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# def normal_init(module, mean=0, std=1, bias=0):
#     """Initialize module parameters with normal distribution."""
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# # --- Activation and Utility Modules (Self-contained or standard PyTorch) ---

# def _make_divisible(v, divisor, min_value=None):
#     """Ensure channel number is divisible by divisor."""
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v

# class swish(nn.Module):
#     """Swish activation function."""
#     def forward(self, x):
#         return x * torch.sigmoid(x)

# class h_swish(nn.Module):
#     """Hard Swish activation function."""
#     def __init__(self, inplace=False):
#         super(h_swish, self).__init__()
#         self.inplace = inplace
#     def forward(self, x):
#         return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

# class h_sigmoid(nn.Module):
#     """Hard Sigmoid activation function."""
#     def __init__(self, inplace=True, h_max=1):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#         self.h_max = h_max # Corresponds to (bias=3.0, divisor=6.0) from original act_cfg
#     def forward(self, x):
#         # Equivalent to ReLU6(x + 3) / 6 when h_max=1
#         return self.relu(x + 3) * self.h_max / 6

# class DYReLU(nn.Module):
#     """Dynamic ReLU implementation."""
#     def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
#                  init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
#         super(DYReLU, self).__init__()
#         self.oup = oup
#         self.lambda_a = lambda_a * 2 # Multiply lambda_a by 2 for scaling adjustments
#         self.K2 = K2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.use_bias = use_bias

#         # Determine the number of parameters needed based on K2 and use_bias
#         if K2: self.exp = 4 if use_bias else 2
#         else: self.exp = 2 if use_bias else 1

#         self.init_a = init_a
#         self.init_b = init_b

#         # Calculate squeeze channels for the bottleneck layer
#         if reduction == 4: squeeze = inp // reduction
#         else: squeeze = _make_divisible(inp // reduction, 4)

#         # Fully connected layers to compute parameters a and b
#         self.fc = nn.Sequential(
#             nn.Linear(inp, squeeze),
#             nn.ReLU(inplace=True),
#             nn.Linear(squeeze, oup * self.exp),
#             h_sigmoid() # Apply Hard Sigmoid to constrain parameters
#         )

#         # Optional spatial attention component
#         if use_spatial:
#             self.spa = nn.Sequential(
#                 nn.Conv2d(inp, 1, kernel_size=1),
#                 nn.BatchNorm2d(1), # Use standard BatchNorm
#             )
#         else:
#             self.spa = None

#     def forward(self, x):
#         # Handle single tensor or list input (legacy?)
#         if isinstance(x, list): x_in, x_out = x[0], x[1]
#         else: x_in, x_out = x, x

#         b, c, h, w = x_in.size()
#         # Global average pooling and parameter generation
#         y = self.avg_pool(x_in).view(b, c)
#         y = self.fc(y).view(b, self.oup * self.exp, 1, 1)

#         # Apply dynamic parameters based on configuration
#         if self.exp == 4: # K2=True, use_bias=True (Piecewise Linear with 2 parts + bias)
#             a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
#             a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
#             b1 = b1 - 0.5 + self.init_b[0]
#             b2 = b2 - 0.5 + self.init_b[1]
#             out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
#         elif self.exp == 2:
#             if self.use_bias: # K2=False, use_bias=True (Linear + bias)
#                 a1, b1 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
#                 b1 = b1 - 0.5 + self.init_b[0]
#                 out = x_out * a1 + b1
#             else: # K2=True, use_bias=False (Piecewise Linear with 2 parts, no bias)
#                 a1, a2 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
#                 a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
#                 out = torch.max(x_out * a1, x_out * a2)
#         elif self.exp == 1: # K2=False, use_bias=False (Linear, no bias)
#             a1 = y
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
#             out = x_out * a1

#         # Apply spatial attention if enabled
#         if self.spa:
#             ys = self.spa(x_in)
#             ys = torch.sigmoid(ys) # Apply sigmoid for gating [0, 1]
#             out = out * ys

#         return out

# # --- Deformable Conv Module Wrapper (using torchvision) ---
# class DyDCNv2(nn.Module):
#     """
#     Modulated Deformable Convolution v2 wrapper using torchvision.ops.DeformConv2d,
#     optionally followed by a normalization layer (GroupNorm supported).

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         stride (int | tuple[int]): Stride of the convolution. Default: 1.
#         norm_cfg (dict | None): Config dict for normalization layer.
#             Example: dict(type='GN', num_groups=16). Default: GroupNorm with 16 groups.
#             Set to None to disable normalization.
#         dcn_kernel_size (int): Kernel size of the deformable convolution. Default: 3.
#         dcn_padding (int): Padding for the deformable convolution. Default: 1.
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  norm_cfg=dict(type='GN', num_groups=16),
#                  dcn_kernel_size=3,
#                  dcn_padding=1):
#         super().__init__()
#         self.with_norm = norm_cfg is not None
#         # Bias is typically disabled if a norm layer follows
#         bias = not self.with_norm

#         # Initialize torchvision's DeformConv2d
#         self.conv = DeformConv2d(
#             in_channels,
#             out_channels,
#             kernel_size=dcn_kernel_size,
#             stride=stride,
#             padding=dcn_padding,
#             bias=bias,
#             # groups=1 # Default groups is 1
#         )

#         # Initialize normalization layer (only GroupNorm implemented here)
#         self.norm = None
#         if self.with_norm:
#             if norm_cfg is None:
#                  pass # No normalization
#             elif norm_cfg['type'] == 'GN':
#                 num_groups = norm_cfg.get('num_groups', 16) # Default to 16 groups if not specified
#                 # affine=True enables learnable scale and shift parameters (like requires_grad=True)
#                 self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
#             else:
#                 # Raise error for unsupported normalization types
#                 raise NotImplementedError(f"Normalization type {norm_cfg['type']} not implemented in DyDCNv2.")

#     def forward(self, x, offset, mask):
#         """
#         Forward pass for DyDCNv2.

#         Args:
#             x (Tensor): Input feature map. Shape: (N, C_in, H_in, W_in).
#             offset (Tensor): Offset tensor for DeformConv2d.
#                              Shape: (N, 2 * K_h * K_w, H_out, W_out).
#             mask (Tensor): Mask tensor for DeformConv2d modulation.
#                            Shape: (N, K_h * K_w, H_out, W_out).
#         """
#         # Apply deformable convolution with separate offset and mask
#         x = self.conv(x.contiguous(), offset=offset, mask=mask)
#         # Apply normalization if configured
#         if self.with_norm and self.norm is not None:
#             x = self.norm(x)
#         return x


# # --- Dynamic Head Block ---
# class DyHeadBlock(nn.Module):
#     """
#     Single block of the Dynamic Head architecture, implementing spatial-aware,
#     scale-aware, and task-aware attention mechanisms using DyDCNv2.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         zero_init_offset (bool): Whether to initialize the offset-generating
#                                  convolution with zeros. Default: True.
#         dcn_kernel_size (int): Kernel size for the deformable convolutions. Default: 3.
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  zero_init_offset=True,
#                  dcn_kernel_size=3):
#         super().__init__()
#         self.zero_init_offset = zero_init_offset
#         kernel_size = dcn_kernel_size
#         dcn_padding = (kernel_size - 1) // 2 # Calculate padding for 'same' spatial size with stride 1

#         # Calculate dimensions for offset and mask based on kernel size
#         # Offset: 2 values (dx, dy) per kernel element
#         self.offset_dim = 2 * kernel_size * kernel_size
#         # Mask: 1 value per kernel element
#         self.mask_dim = kernel_size * kernel_size
#         # Total channels generated by the offset convolution
#         self.offset_and_mask_dim = self.offset_dim + self.mask_dim

#         # Scale-aware attention module (Channel attention using global pooling)
#         self.scale_attn_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(out_channels, 1, kernel_size=1), # 1x1 Conv for channel attention
#             nn.ReLU(inplace=True),
#             h_sigmoid(inplace=True) # Use Hard Sigmoid for gating [0, 1]
#         )

#         # Default normalization configuration (GroupNorm)
#         norm_cfg = dict(type='GN', num_groups=16)

#         # Spatial-aware attention modules (using DyDCNv2)
#         # High-level features (lower resolution input) - stride 1 DCN
#         self.spatial_conv_high = DyDCNv2(in_channels, out_channels, stride=1, norm_cfg=norm_cfg, dcn_kernel_size=kernel_size, dcn_padding=dcn_padding)
#         # Mid-level features (current resolution input) - stride 1 DCN
#         self.spatial_conv_mid = DyDCNv2(in_channels, out_channels, stride=1, norm_cfg=norm_cfg, dcn_kernel_size=kernel_size, dcn_padding=dcn_padding)
#         # Low-level features (higher resolution input) - stride 2 DCN
#         self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2, norm_cfg=norm_cfg, dcn_kernel_size=kernel_size, dcn_padding=dcn_padding)

#         # Convolution layer to generate combined offset and mask
#         self.spatial_conv_offset = nn.Conv2d(
#             in_channels, self.offset_and_mask_dim, kernel_size=kernel_size, padding=dcn_padding)

#         # Task-aware attention module (using DYReLU)
#         self.task_attn_module = DYReLU(out_channels, out_channels)

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         """Initialize weights of the DyHeadBlock."""
#         # Initialize Conv2d layers
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # Use Kaiming Normal initialization for Conv2d layers
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 # Or use the original initialization style if preferred:
#                 normal_init(m, mean=0, std=0.01)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 # Initialize normalization layers
#                  if hasattr(m, 'weight') and m.weight is not None:
#                      nn.init.constant_(m.weight, 1)
#                  if hasattr(m, 'bias') and m.bias is not None:
#                      nn.init.constant_(m.bias, 0)

#         # Zero-initialize the offset-generating convolution if specified
#         if self.zero_init_offset:
#              if hasattr(self, 'spatial_conv_offset') and isinstance(self.spatial_conv_offset, nn.Conv2d):
#                  constant_init(self.spatial_conv_offset, val=0)

#     def forward(self, features):
#         """
#         Forward pass for the DyHeadBlock.

#         Args:
#             features (list[Tensor]): List of feature maps from different levels
#                                      (e.g., [P3, P4, P5] from FPN). Features should
#                                      be ordered from lower level (higher res) to
#                                      higher level (lower res).

#         Returns:
#             list[Tensor]: List of processed feature maps for each level.
#         """
#         outs = []
#         num_levels = len(features)

#         # Iterate through each feature level
#         for level in range(num_levels):
#             # --- Offset/Mask Generation ---
#             # Generate combined offset and mask from the *current* level feature
#             current_feature = features[level]
#             offset_and_mask = self.spatial_conv_offset(current_feature)

#             # Split into separate offset and mask tensors
#             offset = offset_and_mask[:, :self.offset_dim, :, :]
#             # Apply sigmoid to the mask part to constrain values to [0, 1]
#             mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
#             # Current offset/mask spatial size matches current_feature spatial size

#             # --- Spatial-aware Attention ---
#             # 1. Process Mid-level feature (current level)
#             # Offset/mask spatial size matches input/output spatial size here
#             mid_feat = self.spatial_conv_mid(current_feature, offset, mask)

#             # --- Scale-aware Attention (Summation and Gating) ---
#             # Initialize summed feature with the mid-level contribution
#             sum_feat = mid_feat * self.scale_attn_module(mid_feat)
#             summed_levels = 1 # Count the number of levels summed

#             # 2. Fuse from Lower level (if available) - Higher resolution input
#             if level > 0:
#                 lower_feature = features[level - 1]
#                 # spatial_conv_low uses stride=2, so its output spatial size
#                 # will match the current level's spatial size.
#                 # Therefore, the offset/mask generated from current level match
#                 # the expected *output* size of spatial_conv_low. No resize needed.
#                 low_feat = self.spatial_conv_low(lower_feature, offset, mask)
#                 # Add contribution to sum_feat, gated by its own scale attention
#                 sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
#                 summed_levels += 1

#             # 3. Fuse from Higher level (if available) - Lower resolution input
#             if level < num_levels - 1:
#                 higher_feature = features[level + 1]
#                 # spatial_conv_high uses stride=1. Its *output* spatial size
#                 # will match the *input* spatial size (higher_feature).
#                 # The offset/mask were generated from current_feature (larger size).
#                 # We MUST resize offset/mask to match the expected output size.
#                 target_spatial_size = higher_feature.shape[-2:] # H_out, W_out for high conv
#                 offset_resized = F.interpolate(offset, size=target_spatial_size, mode='bilinear', align_corners=False)
#                 mask_resized = F.interpolate(mask, size=target_spatial_size, mode='bilinear', align_corners=False)
#                 # Clamp mask after interpolation just in case (though sigmoid helps)
#                 # mask_resized = torch.clamp(mask_resized, 0, 1)

#                 # Apply DCN to the higher level feature using resized offset/mask
#                 high_feat_processed = self.spatial_conv_high(higher_feature, offset_resized, mask_resized)

#                 # Interpolate the processed feature *back up* to the current level's size
#                 high_feat = F.interpolate(
#                     high_feat_processed,
#                     size=current_feature.shape[-2:], # Target size H, W of current level
#                     mode='bilinear',
#                     align_corners=False) # Use False generally
#                 # Add contribution to sum_feat, gated by its own scale attention
#                 sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
#                 summed_levels += 1

#             # --- Task-aware Attention ---
#             # Average the contributions from different levels
#             # Add epsilon to prevent division by zero, although summed_levels should be >= 1
#             averaged_feat = sum_feat / (summed_levels + 1e-6)
#             # Apply DYReLU for task-aware attention
#             task_aware_feat = self.task_attn_module(averaged_feat)
#             outs.append(task_aware_feat)

#         return outs


# # --- Dynamic Head Neck ---
# class DyHead(nn.Module):
#     """
#     Dynamic Head neck module composed of multiple DyHeadBlocks.

#     Args:
#         in_channels (int | list[int]): Number of input channels. If int, assumes
#                                         same channels for all input levels. If list,
#                                         specifies channels for each level (must match
#                                         input feature list). Currently assumes int.
#         out_channels (int): Number of output channels for each block.
#         num_blocks (int): Number of DyHeadBlocks to stack. Default: 6.
#         zero_init_offset (bool): Whether to initialize offset convs with zeros
#                                  in the blocks. Default: True.
#     """
#     def __init__(self,
#                  in_channels,  # TODO: Handle list[int] if needed by framework
#                  out_channels,
#                  num_blocks=6,
#                  zero_init_offset=True):
#         super().__init__()
#         # Assuming in_channels is an integer for now, representing channels of input features
#         # to the *first* block. Frameworks like mmdet handle varying input channels.
#         # If used standalone or with ultralytics, ensure input features have consistent channels
#         # or modify DyHeadBlock/DyHead to handle varying channels if necessary.
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_blocks = num_blocks

#         dyhead_blocks = []
#         current_in_channels = self.in_channels
#         for i in range(num_blocks):
#             dyhead_blocks.append(
#                 DyHeadBlock(
#                     in_channels=current_in_channels, # Input channels for this block
#                     out_channels=self.out_channels,
#                     zero_init_offset=zero_init_offset
#                     # dcn_kernel_size can be added as param if needed
#                 )
#             )
#             # Input channels for the next block is the output channels of the current one
#             current_in_channels = self.out_channels

#         # Stack the blocks sequentially
#         self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

#     def forward(self, inputs):
#         """
#         Forward pass for the DyHead neck.

#         Args:
#             inputs (tuple[Tensor] | list[Tensor]): List or tuple of feature maps
#                                                    from the backbone/FPN, ordered
#                                                    from lowest level (highest res)
#                                                    to highest level (lowest res).

#         Returns:
#             tuple[Tensor]: Processed feature maps from the final DyHeadBlock,
#                            in the same order as input.
#         """
#         # Ensure input is a list or tuple
#         assert isinstance(inputs, (tuple, list)), \
#             f"Input must be a list or tuple of tensors, got {type(inputs)}"
#         # nn.Sequential handles passing the output of one block to the next
#         # when the input/output is a list/tuple of tensors.
#         outs = self.dyhead_blocks(inputs)
#         # Ensure output is a tuple for consistency with many detection heads
#         if isinstance(outs, list):
#             outs = tuple(outs)
#         return outs

# # --- Example Usage (for standalone testing) ---
# if __name__ == '__main__':
#     print("Running DyHead standalone test...")

#     # Example parameters
#     in_channels_example = 128 # Example input channels from FPN
#     out_channels_example = 128 # Output channels of DyHead blocks
#     num_levels_example = 3     # Example number of FPN levels (e.g., P3, P4, P5)
#     num_dyhead_blocks = 4      # Number of DyHead blocks to stack
#     batch_size_example = 2

#     # Create dummy input feature maps (simulate FPN output)
#     # Ordered from highest resolution (P3) to lowest (P5)
#     input_features_example = [
#         torch.randn(batch_size_example, in_channels_example, 40, 40), # P3 size example
#         torch.randn(batch_size_example, in_channels_example, 20, 20), # P4 size example
#         torch.randn(batch_size_example, in_channels_example, 10, 10), # P5 size example
#     ]
#     print(f"Created {len(input_features_example)} input feature levels.")
#     for i, feat in enumerate(input_features_example):
#         print(f" Input Level {i} shape: {feat.shape}")

#     # Instantiate DyHead
#     dyhead_model = DyHead(
#         in_channels=in_channels_example,
#         out_channels=out_channels_example,
#         num_blocks=num_dyhead_blocks
#     )
#     print(f"\nInstantiated DyHead with {num_dyhead_blocks} blocks.")
#     # print(dyhead_model) # Optional: Print model structure

#     # Perform forward pass
#     print("\nPerforming forward pass...")
#     try:
#         output_features_example = dyhead_model(input_features_example)
#         print("Forward pass successful!")

#         # Check output shapes
#         print("\nOutput feature shapes:")
#         assert isinstance(output_features_example, tuple), "Output should be a tuple"
#         assert len(output_features_example) == num_levels_example, "Number of output levels should match input levels"
#         for i, feat in enumerate(output_features_example):
#             print(f" Output Level {i} shape: {feat.shape}")
#             # Verify output shape matches corresponding input shape spatially
#             # and has the correct number of output channels
#             assert feat.shape[0] == batch_size_example
#             assert feat.shape[1] == out_channels_example
#             assert feat.shape[2] == input_features_example[i].shape[2]
#             assert feat.shape[3] == input_features_example[i].shape[3]
#         print("\nOutput shapes verified successfully.")

#     except Exception as e:
#         print(f"\nERROR during DyHead forward pass: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\nDyHead standalone test finished.")


# Copyright (c) OpenMMLab. All rights reserved.
# Modified to remove mmcv dependencies and temporarily replace DCN with Conv2d for debugging

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    print("torchvision not found or DeformConv2d not available. Install torchvision.")
    print("pip install torchvision")
    # Allow script to continue if only testing the Conv2d replacement
    DeformConv2d = None # Define as None if import fails
import math

# --- Helper Functions ---
def _ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple([x] * n)
    return parse
_pair = _ntuple(2)

# --- Initialization functions ---
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# --- Original Utility Classes (mostly unchanged) ---
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max
    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

class DYReLU(nn.Module):
    # No changes needed here
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_bias = use_bias
        if K2: self.exp = 4 if use_bias else 2
        else: self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b
        if reduction == 4: squeeze = inp // reduction
        else: squeeze = _make_divisible(inp // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(nn.Conv2d(inp, 1, kernel_size=1), nn.BatchNorm2d(1))
        else: self.spa = None

    def forward(self, x):
        if isinstance(x, list): x_in, x_out = x[0], x[1]
        else: x_in, x_out = x, x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1
            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)
        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            out = x_out * a1
        if self.spa:
            ys = self.spa(x_in)
            ys = torch.sigmoid(ys)
            out = out * ys
        return out

# --- DyDCNv2 class (Kept for easy revert, but not used by modified DyHeadBlock) ---
class DyDCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16),
                 dcn_kernel_size=3,
                 dcn_padding=1):
        super().__init__()
        if DeformConv2d is None:
            raise ImportError("torchvision.ops.DeformConv2d is required for DyDCNv2 but not found.")
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = DeformConv2d(
            in_channels, out_channels, kernel_size=dcn_kernel_size,
            stride=stride, padding=dcn_padding, bias=bias
        )
        self.norm = None
        if self.with_norm:
            if norm_cfg is None: pass
            elif norm_cfg['type'] == 'GN':
                num_groups = norm_cfg.get('num_groups', 16)
                self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
            else:
                raise NotImplementedError(f"Normalization type {norm_cfg['type']} not implemented.")

    def forward(self, x, offset, mask):
        x = self.conv(x.contiguous(), offset=offset, mask=mask)
        if self.with_norm and self.norm is not None:
            x = self.norm(x)
        return x


# --- DyHeadBlock (Modified for Debugging - Replaced DCN with Conv2d) ---
class DyHeadBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 zero_init_offset=True, # Kept param, but offset conv is disabled
                 dcn_kernel_size=3):
        super().__init__()
        # print("--- WARNING: Running DyHeadBlock with DCN temporarily replaced by Conv2d for debugging! ---") # Add a warning
        # self.zero_init_offset = zero_init_offset # No longer used directly here

        kernel_size = dcn_kernel_size
        dcn_padding = (kernel_size - 1) // 2

        # Original DCN dims (kept for reference, but not used for offset conv)
        # self.offset_dim = 2 * kernel_size * kernel_size
        # self.mask_dim = kernel_size * kernel_size
        # self.offset_and_mask_dim = self.offset_dim + self.mask_dim

        # Scale-aware attention module (remains the same)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            h_sigmoid(inplace=True)
        )

        # --- TEMPORARY REPLACEMENT: Use Conv2d + GroupNorm instead of DyDCNv2 ---
        norm_cfg = dict(type='GN', num_groups=16) # Default norm config
        use_bias = norm_cfg is None # Set bias=True only if no normalization

        self.spatial_conv_high = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dcn_padding, bias=use_bias)
        self.norm_high = nn.GroupNorm(norm_cfg['num_groups'], out_channels) if norm_cfg else nn.Identity()

        self.spatial_conv_mid = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dcn_padding, bias=use_bias)
        self.norm_mid = nn.GroupNorm(norm_cfg['num_groups'], out_channels) if norm_cfg else nn.Identity()

        self.spatial_conv_low = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=dcn_padding, bias=use_bias)
        self.norm_low = nn.GroupNorm(norm_cfg['num_groups'], out_channels) if norm_cfg else nn.Identity()
        # --- END OF TEMPORARY REPLACEMENT ---

        # --- TEMPORARY REPLACEMENT: Disable offset/mask generation ---
        # self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, kernel_size=kernel_size, padding=dcn_padding)
        self.spatial_conv_offset = nn.Identity() # Replace with Identity, won't be used
        # --- END OF TEMPORARY REPLACEMENT ---

        # Task-aware attention module (remains the same)
        self.task_attn_module = DYReLU(out_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        # Initialize Conv2d and Norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use default init or apply normal_init if preferred
                # normal_init(m, mean=0, std=0.01) # Optional: keep original init style
                pass # Often default init is fine for Conv2d
            elif isinstance(m, nn.GroupNorm):
                 if hasattr(m, 'weight') and m.weight is not None:
                     nn.init.constant_(m.weight, 1)
                 if hasattr(m, 'bias') and m.bias is not None:
                     nn.init.constant_(m.bias, 0)

        # --- TEMPORARY: Offset init not needed ---
        # if self.zero_init_offset:
        #     # Need to check if self.spatial_conv_offset exists if we didn't use Identity
        #     if hasattr(self, 'spatial_conv_offset') and isinstance(self.spatial_conv_offset, nn.Conv2d):
        #          constant_init(self.spatial_conv_offset, val=0)
        # --- END OF TEMPORARY ---


    def forward(self, x):
        """Forward function (Modified for Debugging - Uses Conv2d)."""
        outs = []
        num_levels = len(x)

        for level in range(num_levels):
            # --- TEMPORARY: Disable offset/mask calculation ---
            # offset_and_mask = self.spatial_conv_offset(x[level]) # Disabled
            # offset = offset_and_mask[:, :self.offset_dim, :, :]  # Disabled
            # mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid() # Disabled
            # --- END OF TEMPORARY ---

            # --- Spatial Attention (using Conv2d) ---
            # Apply Conv2d + Norm to the middle level feature
            mid_feat = self.norm_mid(self.spatial_conv_mid(x[level])) # Pass input directly

            # Initial sum
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1

            # Fuse from lower level (higher resolution input x[level-1])
            if level > 0:
                # Apply Conv2d (stride=2) + Norm
                low_feat = self.norm_low(self.spatial_conv_low(x[level - 1])) # Pass input directly
                sum_feat = sum_feat + low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1

            # Fuse from higher level (lower resolution input x[level+1])
            if level < num_levels - 1:
                # --- TEMPORARY: Disable offset/mask resizing ---
                # target_size = x[level + 1].shape[-2:]             # Disabled
                # offset_resized = F.interpolate(offset, size=target_size, mode='bilinear', align_corners=False) # Disabled
                # mask_resized = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False) # Disabled
                # --- END OF TEMPORARY ---

                # Apply Conv2d + Norm to the higher level feature
                high_feat_processed = self.norm_high(self.spatial_conv_high(x[level + 1])) # Pass input directly

                # Upsample the processed feature
                high_feat = F.interpolate(
                    high_feat_processed,
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1

            # Average features and apply task attention
            averaged_feat = sum_feat / summed_levels
            outs.append(self.task_attn_module(averaged_feat))

        return outs


# --- DyHead Class (Unchanged) ---
class DyHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=6,
                 zero_init_offset=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        dyhead_blocks = []
        current_in_channels = self.in_channels
        for i in range(num_blocks):
            dyhead_blocks.append(
                DyHeadBlock( # This will now instantiate the modified block
                    current_in_channels,
                    self.out_channels,
                    zero_init_offset=zero_init_offset)) # Param still passed but maybe unused
            current_in_channels = self.out_channels

        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

    def forward(self, inputs):
        assert isinstance(inputs, (tuple, list)), "Input should be a list or tuple of feature maps"
        outs = self.dyhead_blocks(inputs)
        return outs

# --- Example Usage (Unchanged, will use the modified DyHeadBlock) ---
if __name__ == '__main__':
    in_channels = 256
    out_channels = 256
    num_levels = 4
    batch_size = 2
    feat_sizes = [(batch_size, in_channels, 80, 80), (batch_size, in_channels, 40, 40),
                  (batch_size, in_channels, 20, 20), (batch_size, in_channels, 10, 10)]
    input_features = [torch.randn(s) for s in feat_sizes]
    dyhead = DyHead(in_channels=in_channels, out_channels=out_channels, num_blocks=2)
    print("Running forward pass with DCN replaced by Conv2d...")
    output_features = dyhead(input_features)
    print("Forward pass successful!")
    print("\nInput feature shapes:")
    for i, feat in enumerate(input_features): print(f" Level {i}: {feat.shape}")
    print("\nOutput feature shapes:")
    for i, feat in enumerate(output_features):
        print(f" Level {i}: {feat.shape}")
        assert feat.shape[0] == batch_size
        assert feat.shape[1] == out_channels
        assert feat.shape[2] == input_features[i].shape[2]
        assert feat.shape[3] == input_features[i].shape[3]
    print("\nDyHead (Conv2d replacement) ran successfully with shape checks!")