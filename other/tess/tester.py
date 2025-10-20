import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import objectdata # Import your modified objectdata
from models import get_fasterrcnn_model_single_class as fmodel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import StepLR, LinearLR, SequentialLR

def collate_fn(batch):
    # Filter out None values (e.g., from skipped empty images/frames)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # If batch becomes empty after filtering
        return None, None # Return None for both images and targets
    return tuple(zip(*batch))

def trainer(json_data_path): # Pass the path to the combined JSON file
    
    # Define your class names. These MUST include all unique 'Label' values from your combined JSON data,
    # excluding the '__background__' class which is handled by objectdata internally.
    # Based on your provided my_combined_output.json: "Ball", "Ball1", "Ball2"
    class_names = ['Ball', 'Ball1', 'Ball2']  # Adjusted to include all possible labels.
    num_classes = len(class_names) + 1 # +1 for background class

    # Define your transformations for training
    transform = A.Compose([
        A.Resize(640, 640), # All images and bboxes will be scaled to 640x640
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet stds
        ToTensorV2() 
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # Specify format for bboxes

    # Create the dataset
    # The objectdata class now handles reading the JSON and determining image paths internally.
    # It also handles knowing the source dimensions of the bboxes.
    # Corrected call to objectdata:
    train_dataset = objectdata(json_data_path, class_names, transform=transform)

    # Create the DataLoader
    train_dataloader = DataLoader(
        train_dataset, # Use the correctly named dataset variable
        batch_size=4, # Adjust batch size as needed
        shuffle=True, # Shuffle for training
        num_workers=0, # Set to 0 for Windows, >0 for Linux/macOS for faster loading
        collate_fn=collate_fn # Use the custom collate function for object detection
    )

    # Load your Faster R-CNN model
    model = fmodel(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(train_dataloader) - 1) 

    lr_scheduler_warmup = LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    
    lr_scheduler_main = StepLR(optimizer, step_size=3, gamma=0.1)

    lr_scheduler = SequentialLR(optimizer, schedulers=[lr_scheduler_warmup, lr_scheduler_main], milestones=[warmup_iters])


    num_epochs = 10000 # Define number of training epochs

    print(f"Starting training on device: {device}")
    print(f"Total batches per epoch: {len(train_dataloader)}")
    print(f"Total dataset size: {len(train_dataset)}") # Corrected to use train_dataset

    # Training loop
    model.train() # Set the model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            if images is None and targets is None: # Handle empty batch after collate_fn filtering
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_dataloader)}, Skipping empty batch.")
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if batch_idx % 10 == 0: # Print loss every 10 batches
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {losses.item():.4f}")

        lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    torch.save(model.state_dict(), 'trained_model_final.pth')
    print("Training finished and final model saved!")

if __name__ == "__main__":
    combined_json_dataset_path = "my_combined_output.json" # Path to your combined JSON file

    if os.path.exists(combined_json_dataset_path):
        # The `trainer` function now expects the path to the JSON file directly
        trainer(combined_json_dataset_path) 
    else:
        print(f"Error: Combined JSON dataset file not found at {combined_json_dataset_path}. Please ensure it exists.")