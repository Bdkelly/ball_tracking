import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model_single_class(num_classes=2):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
<<<<<<< HEAD
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
=======

    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
>>>>>>> fcb97226bc8bf359259302cf073874eb5f601f3e
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model