# Tệp: utils/accur_eiou_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AccurEIOU_Loss(nn.Module):
    """
    AccurEIOU Loss function cho NSEC-YOLO.
    Kết hợp ưu điểm của CIOU và EIOU loss cho dự đoán bounding box chính xác hơn.
    """
    def __init__(self, eps=1e-9, half_pi_range=180.0):
        super(AccurEIOU_Loss, self).__init__()
        self.eps = eps
        self.half_pi_range = half_pi_range
        self.threshold = 0.01  # Ngưỡng để xác định khi tỷ lệ khung hình đã ổn định
        self.delta = 0.7    # Weighting factor cho aspect ratio discrepancy
        self.theta = 0.3    # Dynamic adjustment factor
    
    def forward(self, pred, target, reduction='mean'):
        """
        Tính toán AccurEIOU Loss giữa predicted và target bounding boxes.
        
        Args:
            pred: Tensor [batch_size, 4] của predicted boxes (x1, y1, x2, y2)
            target: Tensor [batch_size, 4] của target boxes (x1, y1, x2, y2)
            reduction: Phương pháp giảm thiểu loss ('none', 'mean', 'sum')
            
        Returns:
            Loss tensor
        """
        # Tính toán IoU
        iou, union = self._iou(pred, target)
        
        # Tính toán tọa độ tâm của boxes
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        
        # Tính toán kích thước của boxes
        pred_size = pred[:, 2:] - pred[:, :2]
        target_size = target[:, 2:] - target[:, :2]
        
        # Tính toán width và height riêng lẻ
        pred_w, pred_h = pred_size[:, 0], pred_size[:, 1]
        target_w, target_h = target_size[:, 0], target_size[:, 1]
        
        # 1. Tính toán smallest enclosing box
        c_x1 = torch.min(pred[:, 0], target[:, 0])
        c_y1 = torch.min(pred[:, 1], target[:, 1])
        c_x2 = torch.max(pred[:, 2], target[:, 2])
        c_y2 = torch.max(pred[:, 3], target[:, 3])
        
        # Kích thước của enclosing box
        c_w = c_x2 - c_x1 + self.eps
        c_h = c_y2 - c_y1 + self.eps
        c_area = c_w * c_h + self.eps
        
        # 2. Tính toán khoảng cách giữa tâm (distance loss)
        center_dist_squared = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # 3. Tính toán aspect ratio consistency (CIOU component)
        v = (4 / math.pi**2) * (torch.atan(target_w / (target_h + self.eps)) 
                              - torch.atan(pred_w / (pred_h + self.eps))) ** 2
        alpha = v / (1 - iou + v + self.eps)
        
        # 4. Tính toán width và height consistency (EIOU component)
        w_dist_squared = (pred_w - target_w) ** 2
        h_dist_squared = (pred_h - target_h) ** 2
        
        # 5. Tính loss cho từng thành phần
        loss_iou = 1 - iou
        loss_dist = center_dist_squared / (c_area + self.eps)
        loss_ar_ciou = alpha * v
        loss_ar_eiou = (w_dist_squared / (c_w**2 + self.eps)) + (h_dist_squared / (c_h**2 + self.eps))
        
        # 6. Tính aspect ratio difference
        ar_pred = pred_w / (pred_h + self.eps)
        ar_target = target_w / (target_h + self.eps)
        ar_diff = torch.abs(ar_pred - ar_target)
        
        # 7. Dynamic adjustment dựa trên iteration và aspect ratio difference
        # Trong thực tế, t (iteration count) sẽ được truyền vào từ bên ngoài
        t = 1.0  # placeholder, should be replaced with actual iteration count
        ar_change = self.delta * ar_diff + self.theta * (1 / t)
        
        # 8. Select loss component dựa trên aspect ratio stability
        is_stable = ar_change < self.threshold
        
        # Kết hợp loss
        loss = loss_iou + loss_dist
        loss += torch.where(is_stable, loss_ar_eiou, loss_ar_ciou)
        
        # Áp dụng reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _iou(self, pred, target):
        """
        Tính intersection over union giữa predicted và target boxes.
        
        Args:
            pred: Tensor [batch_size, 4] của predicted boxes (x1, y1, x2, y2)
            target: Tensor [batch_size, 4] của target boxes (x1, y1, x2, y2)
            
        Returns:
            Tuple (iou, union): IoU values và union areas
        """
        # Tính intersection
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])
        
        # Clamp để đảm bảo intersection không âm
        width = (x2 - x1).clamp(min=0)
        height = (y2 - y1).clamp(min=0)
        
        intersection = width * height
        
        # Tính area cho pred và target
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Tính union
        union = pred_area + target_area - intersection + self.eps
        
        # Tính IoU
        iou = intersection / union
        
        return iou, union