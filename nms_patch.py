import torch
import torchvision.ops
import nms_patch
def custom_nms(boxes, scores, iou_threshold):
    """
    Custom NMS implementation that doesn't rely on torchvision C++ ops
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Sort by scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        # Calculate intersection
        x1 = torch.max(current_box[0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[3], remaining_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        
        # Calculate areas
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        
        # Calculate union and IoU
        union = area_current + area_remaining - intersection
        iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero
        
        # Keep boxes with IoU less than threshold
        mask = iou < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# Monkey patch the torchvision NMS
torchvision.ops.nms = custom_nms
print("✓ NMS patched successfully")