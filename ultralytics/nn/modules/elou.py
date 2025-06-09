# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
DetectEloU module for YOLO-CXR with EloU loss integration.
Reference: https://ieeexplore.ieee.org/document/10720017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Detect  # Import base Detect class

class DetectEloU(Detect):
    """
    DetectEloU head for YOLO-CXR.
    Extends standard Detect class with integrated EloU (Enhanced IoU) loss calculation.
    Based on YOLO-CXR paper: https://ieeexplore.ieee.org/document/10720017
    
    The EloU formula enhances standard IoU by adding terms that help with convergence
    and performance, especially for small objects like in X-ray anomaly detection.
    """
    
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # Initialize the parent class (standard Detect)
        self.use_elou = True  # Flag to enable/disable EloU loss

    def loss(self, preds, batch):
        """
        Modified loss function to use EloU (Enhanced IoU) loss.
        
        Args:
            preds (list): Model predictions
            batch (dict): Batch data and targets
        
        Returns:
            dict: Loss components
        """
        # First call the standard loss calculation from parent class
        loss_dict = super().loss(preds, batch)
        
        if not self.use_elou:
            return loss_dict  # If EloU is disabled, return standard loss
            
        # Get required values from loss_dict for EloU calculation
        loss_box = loss_dict.get('box_loss', None)
        
        # Extract prediction and target boxes from the batch
        # Note: This assumes the parent class's loss method has already
        # calculated predicted and target boxes and IoU between them.
        # We need to access these values, structure may vary based on YOLOv8 implementation
        
        # Implementation of EloU loss based on the paper's description
        # Note: This is an interpretation; adjust the formula as needed based on the paper
        if hasattr(self, 'bbox_iou') and hasattr(self, 'pred_bboxes') and hasattr(self, 'target_bboxes'):
            # Access IoU values and bounding boxes if they're available from parent class
            ious = self.bbox_iou  # This should be calculated in parent's loss()
            pred_boxes = self.pred_bboxes
            target_boxes = self.target_bboxes
            
            # Calculate EloU components
            # Extract box coordinates
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.chunk(4, -1)
            target_x1, target_y1, target_x2, target_y2 = target_boxes.chunk(4, -1)
            
            # Calculate width and height
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
            
            # Calculate centers
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            
            # Center distance
            center_distance = torch.sqrt((pred_cx - target_cx)**2 + (pred_cy - target_cy)**2)
            
            # Diagonal distance of the bounding box
            diagonal = torch.sqrt(target_w**2 + target_h**2)
            
            # EloU formula components (adapted from paper)
            # This is an example formulation - adjust based on the exact formula from the paper
            alpha = 0.25  # Hyperparameter
            beta = 0.5    # Hyperparameter
            gamma = 1.0   # Hyperparameter
            
            # Center point distance penalty term
            center_term = alpha * center_distance / diagonal
            
            # Aspect ratio consistency term
            ar_term = beta * torch.abs(torch.atan(target_w/target_h) - torch.atan(pred_w/pred_h)) / (torch.pi/4)
            
            # Scale consistency term
            scale_term = gamma * torch.abs(pred_w * pred_h - target_w * target_h) / (target_w * target_h)
            
            # Combine with standard IoU
            elou_loss = (1 - ious) + center_term + ar_term + scale_term
            
            # Replace standard box_loss with EloU loss
            loss_dict['box_loss'] = elou_loss.mean()
            # Add EloU components for logging
            loss_dict['elou_loss'] = elou_loss.mean().detach()
            loss_dict['center_term'] = center_term.mean().detach()
            loss_dict['ar_term'] = ar_term.mean().detach()
            loss_dict['scale_term'] = scale_term.mean().detach()
            
        else:
            # Fallback if we can't access the required values
            print("WARNING: Unable to apply EloU loss, required attributes not found. Using standard loss.")
        
        return loss_dict