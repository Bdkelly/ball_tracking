from datasets import objectdata
import os
from jsonreader import infbondbox
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from PIL import Image
import json
import torch

def collate_fn(batch):
    return tuple(zip(*batch))

def getvals(frame):
    with Image.open(frame) as img:
        wid,hei = img.size
        print(f"{wid},{hei}")

def getFirst(data,img_dir):
    fimg = f"{list(data[0])[0]}"
    first_img = f"{img_dir}/{fimg}"
    if fimg.endswith((".png",".jpg")):
        ftype = ""
    else:    
        if os.path.exists(f"{first_img}.png"):
            ftype = ".png"
        else:
            ftype = ".jpg"
    first_img = first_img + ftype
    print(first_img)

if __name__ == "__main__":
    box2 = "/Users/Ben/Documents/dever/python/ptorch/data/video1p.json"
    img2 = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/video1/imgs"
    box1 = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/inferred_images/mlsvideo/jdata/cleaned_json/comb/vb.json"
    img1 = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/mlsvideo/imgs"
    
    data = infbondbox(box1)
    for i in data:
        print(i)
    print(len(data))
    getFirst(data,img1)


    
