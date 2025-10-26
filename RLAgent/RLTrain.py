import torch
import albumentations as A
from utils.models import get_fasterrcnn_model_single_class as fmodel
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']