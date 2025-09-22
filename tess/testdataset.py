import torch
from jsonreader import extbondbox
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np

class objectdata(Dataset):
    def __init__(self, json_data_path, class_names, transform=None):
        # extbondbox now returns a list of dictionaries, each containing 'image_file_path' and 'detections'
        self.all_annotated_frames = json_data_path 

        if not self.all_annotated_frames:
            raise ValueError(f"No valid annotations found in {json_data_path}. Dataset cannot be created or is empty.")

        self.class_names = ['__background__'] + class_names 
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}

        # The bounding box coordinates in your JSON are natively pixel values
        # derived from a 1920x1080 source resolution. This is their "native" pixel space.
        self.json_bbox_native_width = 1920 
        self.json_bbox_native_height = 1080

        # We will dynamically get the actual image dimensions in __getitem__
        # to ensure correct scaling for Albumentations.

        self.transform = transform

    def __len__(self):
        return len(self.all_annotated_frames)

    def __getitem__(self, idx):
        frame_data_entry = self.all_annotated_frames[idx]
        image_file_path = frame_data_entry["image_file_path"]
        detections = frame_data_entry["detections"]

        image = Image.open(image_file_path).convert("RGB")
        image_np = np.array(image) # Convert PIL image to NumPy array

        # Get the actual dimensions of the loaded image
        actual_image_height, actual_image_width = image_np.shape[:2]

        boxes = []
        labels = []

        # Calculate scaling factors from JSON's native bbox space to the actual image's dimensions
        scale_x = actual_image_width / self.json_bbox_native_width
        scale_y = actual_image_height / self.json_bbox_native_height

        for det in detections:
            # Detections from JSON are already x_min, y_min, x_max, y_max (pixel coordinates relative to 1920x1080)
            x_min_json, y_min_json, x_max_json, y_max_json = det['x_min'], det['y_min'], det['x_max'], det['y_max']
            label_name = det['Label']

            # Scale JSON coordinates to the actual loaded image dimensions
            x_min_scaled = x_min_json * scale_x
            y_min_scaled = y_min_json * scale_y
            x_max_scaled = x_max_json * scale_x
            y_max_scaled = y_max_json * scale_y

            # Convert to int and clip to be within the actual image bounds
            # Rounding is important before converting to int
            x_min_scaled = int(round(max(0.0, x_min_scaled)))
            y_min_scaled = int(round(max(0.0, y_min_scaled)))
            x_max_scaled = int(round(min(float(actual_image_width), x_max_scaled)))
            y_max_scaled = int(round(min(float(actual_image_height), y_max_scaled)))
            
            # Check for degenerate boxes *after* scaling and clipping
            if x_max_scaled <= x_min_scaled or y_max_scaled <= y_min_scaled:
                print(f"WARNING: Degenerate box for frame {frame_data_entry['frame_key']} after scaling to {actual_image_width}x{actual_image_height}. Skipping. Original JSON: ({x_min_json},{y_min_json},{x_max_json},{y_max_json}) -> Scaled: ({x_min_scaled},{y_min_scaled},{x_max_scaled},{y_max_scaled})")
                continue

            boxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
            labels.append(self.label_to_id.get(label_name, 0)) # Default to 0 (background) if label not found

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # Ensure image_np has valid shape before passing to transform
        if actual_image_height == 0 or actual_image_width == 0:
            print(f"Warning: Image at {image_file_path} has zero dimensions. Returning dummy data.")
            # Return dummy tensors if image is invalid
            return torch.zeros((3, 640, 640), dtype=torch.float32), {'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty((0,), dtype=torch.int64), 'image_id': torch.tensor([idx]), 'area': torch.empty((0,), dtype=torch.float32), 'iscrowd': torch.empty((0,), dtype=torch.int64)}


        if self.transform:
            # Albumentations now receives pixel coordinates that are correctly scaled
            # to the `image_np`'s dimensions.
            transformed = self.transform(
                image=image_np, 
                bboxes=boxes_tensor.tolist(), # Pass as list of lists for Albumentations
                labels=labels_tensor.tolist(), # Pass as list of ints
                rows=actual_image_height,      # Tell Albumentations the current image height
                cols=actual_image_width        # Tell Albumentations the current image width
            )
            image_transformed = transformed['image']
            boxes_transformed = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels_transformed = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image_transformed = ToTensorV2()(image=image_np)['image']
            boxes_transformed = boxes_tensor
            labels_transformed = labels_tensor

        target = {}
        target["boxes"] = boxes_transformed
        target["labels"] = labels_transformed
        target["image_id"] = torch.tensor([idx])
        print(target)
        
        if boxes_transformed.numel() > 0:
            target["area"] = (boxes_transformed[:, 3] - boxes_transformed[:, 1]) * (boxes_transformed[:, 2] - boxes_transformed[:, 0])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        target["iscrowd"] = torch.zeros((len(boxes_transformed),), dtype=torch.int64)

        return image_transformed, target
        
        

if __name__ == "__main__":
    # IMPORTANT: Adjust these paths to your actual setup
    # This should be the directory containing your extracted 'frame_00XXX.jpg' images
    videodir = "/video1"
    basepath = r"/Users/Ben/Documents/dever/python/ptorch/data/outframes"
    jsonpath = "my_combined_output.json"
    img_dir = basepath+videodir+"/imgs/"
    parsed_json_annotations = extbondbox(jsonpath)
    # parsed_json_annotations = 
    # These class names MUST match the 'labels' used in your Label Studio annotations exactly.
    # Exclude '__background__', as it's added automatically by the dataset class.
    CLASS_NAMES_IN_PROJECT = ['Ball'] 

    # 1. Use extbondbox to parse the JSON and get the annotations dictionary

    if not parsed_json_annotations:
        print("Exiting: No annotations parsed from JSON file.")
        exit()

    # 2. Define your Albumentations transform pipeline for training.
    # This should match what you use in train.py exactly (including Resize, Normalize).
    training_transform_pipeline = A.Compose([
        A.Resize(640, 640), # Resize images to model input size
        A.HorizontalFlip(p=0.5), # Example augmentation
        A.RandomBrightnessContrast(p=0.2), # Example augmentation
        A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet stds
        ToTensorV2() # Converts image to PyTorch tensor (HWC to CHW)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # Crucial for bounding box transforms

    # 3. Create the dataset instance, passing the parsed annotations
    if os.path.exists(img_dir):
        try:
            dataset = objectdata(
                json_data_path=parsed_json_annotations, # Pass the parsed dict here
                class_names=CLASS_NAMES_IN_PROJECT, # <--- THIS IS THE MISSING ARGUMENT YOU NEEDED TO ADD
                transform=training_transform_pipeline # Pass your training transforms
            )
            
            # 4. Create the DataLoader
            # num_workers > 0 requires the main script to be under if __name__ == "__main__" on Windows
            # For testing, 0 is fine. For actual training performance, use a higher number (e.g., 4 or os.cpu_count() - 1).
            dataloader = DataLoader(
                dataset, 
                batch_size=2, # Use a small batch size for testing
                shuffle=True, 
                num_workers=0, # Set to 0 for initial debugging to avoid multiprocessing issues
                collate_fn=lambda x: tuple(zip(*x)) # Custom collate_fn for object detection
            ) 

            print(f"Dataset has {len(dataset)} annotated frames prepared for loading.")
            
            # 5. Iterate through a few batches to verify the output
            print("\n--- Verifying DataLoader output ---")
            for i, (images, targets) in enumerate(dataloader):
                print(f"\nBatch {i+1}:")
                print(f"  Number of images in batch: {len(images)}")
                print(f"  Number of targets in batch: {len(targets)}")

                if len(images) > 0:
                    print(f"  First image tensor shape: {images[0].shape}")
                
                if len(targets) > 0:
                    for j, target_dict in enumerate(targets):
                        print(f"  Target {j+1} (for image {j+1}) keys: {target_dict.keys()}")
                        if 'boxes' in target_dict and target_dict['boxes'].numel() > 0:
                            print(f"    Boxes shape: {target_dict['boxes'].shape}, Labels shape: {target_dict['labels'].shape}")
                            # Print first box's coordinates to check scaling and format
                            print(f"    First box content (pixel coords, after transforms): {target_dict['boxes'][0].tolist()}")
                            print(f"    First label content: {target_dict['labels'][0].item()} (raw ID)")
                        else:
                            print(f"    No bounding boxes found for this image (after filtering/processing).")

                if i >= 2: # Process 3 batches for testing
                    break

        except Exception as e:
            print(f"An error occurred during dataset/dataloader initialization or iteration: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for deeper debugging
    else:
        print(f"Error: Images directory does not exist at {img_dir}. Please check the path.")