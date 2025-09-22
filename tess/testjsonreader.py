import json
import os 
from PIL import Image # Keeping this import as it might be used for image inspection in other scripts.

def extbondbox(json_data_path):
    """
    Extracts bounding box annotations and associates them with their full image paths
    from a JSON file with interleaved video metadata and frame data (like my_combined_output.json).

    Args:
        json_data_path (str): Path to the input JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary represents an annotated frame
              and includes its full image file path and its detections.
              Example: [
                  {"image_file_path": "/path/to/video1/imgs/frame_00000.jpg", "frame_key": "frame_00000.jpg", "detections": [...]},
                  {"image_file_path": "/path/to/mlsvideo/imgs/frame_00435.jpg", "frame_key": "frame_00435.jpg", "detections": [...]},
                  ...
              ]
    """
    if not os.path.exists(json_data_path):
        print(f"Error: JSON file not found at {json_data_path}")
        return []

    with open(json_data_path, 'r') as f:
        data = json.load(f)

    all_frames_with_paths_and_detections = []
    current_video_base_path = None # To store the image base directory for the currently active video source

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dictionary item in top-level list: {item}")
                continue

            # Check if it's a "videos" metadata block
            if "videos" in item and isinstance(item["videos"], dict):
                # Assuming there's one primary video entry per "videos" block for simplicity.
                # If there are multiple video sources defined in ONE "videos" block,
                # this would need to loop through item["videos"].items() and store a mapping.
                # For `my_combined_output.json` structure, it's typically one video ID per block.
                for video_id, video_info in item["videos"].items():
                    if "path" in video_info:
                        current_video_base_path = video_info["path"] # Update the active image base path
                        print(f"DEBUG: Switched active video base path to: {current_video_base_path} (for video '{video_id}')")
                        # We only need the path here for image loading. Other metadata is not used by datasets.py from here.
                    else:
                        print(f"Warning: 'path' not found for video '{video_id}' in metadata: {video_info}. Frames for this video may be unlocatable.")
                
            # Check if it's a frame annotation block (e.g., {"frame_00000.jpg": [...]})
            elif len(item) == 1 and list(item.keys())[0].startswith("frame_") and list(item.keys())[0].endswith(".jpg"):
                frame_key = list(item.keys())[0]
                detections = item[frame_key]

                if current_video_base_path is None:
                    print(f"Warning: Frame {frame_key} encountered without a preceding video base path. Skipping frame.")
                    continue
                
                # Construct the full image file path. The frame_key already contains the filename.
                image_file_path = f"{current_video_base_path}/{frame_key}"

                # Append to the list, associating the detections with the full image path
                all_frames_with_paths_and_detections.append({
                    "image_file_path": image_file_path,
                    "frame_key": frame_key, # Keep original frame key if needed for other purposes
                    "detections": detections
                })
                # print(f"DEBUG: Processed frame {frame_key} with {len(detections)} detections.")
            else:
                print(f"Warning: Skipping unrecognized top-level item format: {item}")

    print(f"DEBUG: Finished parsing. Total frames extracted with paths: {len(all_frames_with_paths_and_detections)}")
    return all_frames_with_paths_and_detections

# --- Test the extbondbox function ---
if __name__ == "__main__":
    test_json_path = "my_combined_output.json" # Path to your combined JSON file
    
    # Create a dummy my_combined_output.json for testing if it doesn't exist
    if not os.path.exists(test_json_path):
        print(f"Creating a dummy '{test_json_path}' for testing...")
        dummy_data = [
            {"videos": {"video1": {"format": "mp4", "path": "/tmp/video_frames/video1"}}},
            {"frame_00000.jpg": [{"Label": "Ball", "x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20}]},
            {"frame_00001.jpg": [{"Label": "Ball", "x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}]},
            {"videos": {"mlsvideo": {"format": "mp4", "path": "/tmp/video_frames/mlsvideo"}}},
            {"frame_00000.jpg": [{"Label": "Ball1", "x_min": 100, "y_min": 100, "x_max": 110, "y_max": 110}]}, # Duplicate frame num, different path
            {"frame_00005.jpg": [{"Label": "Ball1", "x_min": 150, "y_min": 150, "x_max": 160, "y_max": 160}]},
        ]
        with open(test_json_path, 'w') as f: json.dump(dummy_data, f, indent=2)

        # Create dummy image directories for the test
        os.makedirs("/tmp/video_frames/video1", exist_ok=True)
        os.makedirs("/tmp/video_frames/mlsvideo", exist_ok=True)
        from PIL import Image, ImageDraw # Re-import here for local use
        dummy_img = Image.new('RGB', (1920, 1080), color='blue')
        dummy_img.save("/tmp/video_frames/video1/frame_00000.jpg")
        dummy_img.save("/tmp/video_frames/video1/frame_00001.jpg")
        dummy_img.save("/tmp/video_frames/mlsvideo/frame_00000.jpg")
        dummy_img.save("/tmp/video_frames/mlsvideo/frame_00005.jpg")
        print("DEBUG: Created dummy image files for testing.")


    # Get the list of annotations with full paths
    extracted_data = extbondbox(test_json_path)
    print(f"\n--- Extracted Data Summary ({len(extracted_data)} entries): ---")
    for i, entry in enumerate(extracted_data):
        print(f"Entry {i}: Frame: {entry['frame_key']}, Detections: {len(entry['detections'])}, Image Path: {entry['image_file_path']}")
        if i >= 5: # Print first 6 entries for brevity
            print("...")
            break
    print("--- End of extbondbox test ---")
