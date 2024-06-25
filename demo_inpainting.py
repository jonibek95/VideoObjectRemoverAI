import cv2
import numpy as np
import argparse
import torch
import torchvision
from PIL import Image
import psutil
import os
from track_anything import TrackingAnything, parse_augment
from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor, SahiAutoSegmentation, sahi_sliced_predict

def get_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    # frames = []
    # try:
    #     cap = cv2.VideoCapture(video_path)
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if ret:
    #             # Determine the scaling factor, maintaining aspect ratio
    #             height, width = frame.shape[:2]
    #             max_dimension = max(height, width)
    #             scale = 640.0 / max_dimension if max_dimension > 640 else 1

    #             # Resize the frame, maintaining aspect ratio
    #             if scale != 1:
    #                 frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    #             # Convert color from BGR to RGB
    #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frames.append(frame_rgb)

    #             # Check memory usage and stop if too high
    #             if psutil.virtual_memory().percent > 90:
    #                 print("Memory usage is too high (>90%). Stopping video extraction.")
    #                 break
    #         else:
    #             break
    # except Exception as e:
    #     print(f"Error reading video source {video_path}: {e}")
    # finally:
    #     # Always release the video capture
    #     cap.release()
    return frames

def process_frame_with_sahi(frame_path, sam_model_type, detection_model_path, output_mask_dir, frame_index):
    
    os.makedirs(output_mask_dir, exist_ok=True)
    
    boxes = sahi_sliced_predict(
        image_path=frame_path,
        detection_model_type="yolov8",
        detection_model_path=detection_model_path,
        conf_th=0.5,
        image_size=640,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    sahi_segmentation = SahiAutoSegmentation()
    mask_image = sahi_segmentation.predict(
        source=frame_path,
        model_type=sam_model_type,
        input_box=boxes,
        multimask_output=False,
        random_color=False,
        show=False,
        save=False
    )
    frame = cv2.imread(frame_path)
    
    if isinstance(mask_image, torch.Tensor):
        mask_image_np = mask_image.cpu().numpy()
    else:
        mask_image_np = np.array(mask_image)

    # Ensure mask is correctly formed
    if len(mask_image_np.shape) == 4:
        # Flatten the first two dimensions if necessary
        mask_np = mask_image_np.reshape(-1, mask_image_np.shape[2], mask_image_np.shape[3])

        # You might need to adjust this logic based on how your masks are structured
        mask_np = mask_np[0]  # Taking the first mask as an example
    elif len(mask_image_np.shape) == 3:
        mask_np = mask_image_np[0]
    else:
        print(f"Unexpected mask shape: {mask_image_np.shape}")
        mask_np = np.zeros_like(frame, dtype=np.uint8)

    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8) * 255

    # mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_np

def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")

def main():
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--output_path', default='./output.mp4')
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--points_per_side', type=int, default=4)
    parser.add_argument('--points_per_batch', type=int, default=2)
    parser.add_argument('--min_area', type=int, default=500)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = TrackingAnything(
        "./checkpoints/sam_vit_h_4b8939.pth",
        "./checkpoints/XMem-s012.pth",
        "./checkpoints/E2FGVI-HQ-CVPR22.pth",
        args
    )
    
    frames = get_frames_from_video(args.video_path)
    temp_frames_dir = './output/frames'
    temp_masks_dir = './output/masks'
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(temp_masks_dir, exist_ok=True)

    frames_list = []
    masks_list = []

    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:03d}.png')
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        cv2.imwrite(frame_path, bgr_frame)
        # frame_path = os.path.join(temp_frames_dir, f'frame_{i:03d}.npy')
        # np.save(frame_path, frame)  # Save frame as .npy
        mask_np = process_frame_with_sahi(
            frame_path=frame_path,
            sam_model_type=args.sam_model_type,
            detection_model_path="./checkpoints/yolov8x.pt",
            output_mask_dir=temp_masks_dir,
            frame_index=i
        )
        mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        frames_list.append(frame)
        masks_list.append(mask_np)
        
        mask_filename = f"mask_{i:03d}.png"
        cv2.imwrite(os.path.join(temp_masks_dir, mask_filename), mask_np)
        
        
    
    frames_np = np.array(frames_list).astype(np.float32) / 255.0
    masks_np = np.array(masks_list).astype(np.float32) / 255.0
    masks_np[masks_np > 0] = 1

    # Debug: Check if masks and frames are processed correctly
    print("Debug: Checking masks and frames")
    print("Frames shape:", frames_np.shape)
    print("Masks shape:", masks_np.shape)

    # Perform inpainting with improved parameters.
    try:
        inpainted_frames = model.baseinpainter.inpaint(frames_np, masks_np, ratio=0.8, dilate_radius=3)
        if inpainted_frames is None or not np.any(inpainted_frames):
            print("Inpainting produced no or empty output.")
            raise Exception("Inpainting failed to modify frames.")
    except Exception as e:
        print(f"Inpainting error: {e}")
        # Fallback to original frames if inpainting fails
        inpainted_frames = frames_np

    # Save inpainted frames
    save_path = './output/inpainted_frames'
    os.makedirs(save_path, exist_ok=True)

    for i, inpainted_frame in enumerate(inpainted_frames):
        inpainted_frame_uint8 = (inpainted_frame * 255).astype(np.uint8)
        Image.fromarray(inpainted_frame_uint8).save(os.path.join(save_path, f'inpainted_frame_{i:03d}.png'))

    # Generate video from inpainted frames
    generate_video_from_frames(inpainted_frames, args.output_path, args.fps)
    print(f"Inpainted frames saved to {save_path}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

