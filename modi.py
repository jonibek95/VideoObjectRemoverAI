import cv2
import numpy as np
import argparse
import torch
import torchvision
from PIL import Image
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
    return frames

def generate_video_from_frames(frames, output_path, fps=30):
    # Convert list of numpy arrays to a single numpy array
    frames_np = np.stack(frames, axis=0)  

    # Ensure the array is in the correct shape (T, H, W, C)
    frames_np = frames_np.reshape(-1, 720, 1280, 3) 

    # Convert to tensor
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # Shape becomes (T, C, H, W)

    # Write the video
    torchvision.io.write_video(output_path, frames_tensor, fps=fps, video_codec="libx264")

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
    if frame is None:
        raise ValueError(f"Unable to load frame from {frame_path}")

    if isinstance(mask_image, torch.Tensor):
        mask_np = mask_image.cpu().numpy()
    else:
        mask_np = np.array(mask_image)

    if len(mask_np.shape) == 4 and mask_np.shape[0] > 1:
        mask_np = mask_np[0, 0]
    elif len(mask_np.shape) == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8) * 255

    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
    mask_filename = f"mask_{frame_index:03d}.png"
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask_resized)
    return mask_resized

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

    processed_frames = []

    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:03d}.png')
        cv2.imwrite(frame_path, frame)

        mask_np = process_frame_with_sahi(
            frame_path=frame_path,
            sam_model_type=args.sam_model_type,
            detection_model_path="./checkpoints/yolov8x.pt",
            output_mask_dir=temp_masks_dir,
            frame_index=i
        )

        if mask_np.shape != frame.shape[:2]:
            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

        # Normalize frame and mask
        frame_np = frame.astype(np.float32) / 255.0
        mask_np = mask_np.astype(np.float32) / 255.0

        # Add a channel dimension to mask_np and repeat it 3 times
        frame_np = frame_np.reshape(1, 720, 1280, 3)
        mask_np = mask_np.reshape(1, 720, 1280)
        # mask_np = mask_np[:, :, np.newaxis]
        # mask_np = np.repeat(mask_np, 1, axis=0)
        
        print("frames: ", frame_np.shape)
        print("masks: ", mask_np.shape)

        # Perform inpainting using NumPy arrays
        inpainted_frame_np = model.baseinpainter.inpaint(frame_np, mask_np, ratio=1)

        # No need to convert back to tensor for saving
        processed_frames.append((inpainted_frame_np * 255).astype(np.uint8))

    # Ensure the output path has the correct format, e.g., '.mp4'
    args.output_path = args.output_path if args.output_path.endswith('.mp4') else args.output_path + '.mp4'

    generate_video_from_frames(processed_frames, args.output_path, args.fps)
    print(f"Output video saved to {args.output_path}")
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
import cv2
import numpy as np
import argparse
import torch
import torchvision
from PIL import Image
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
    if frame is None:
        raise ValueError(f"Unable to load frame from {frame_path}")

    if isinstance(mask_image, torch.Tensor):
        mask_np = mask_image.cpu().numpy()
    else:
        mask_np = np.array(mask_image)

    if len(mask_np.shape) == 4 and mask_np.shape[0] > 1:
        mask_np = mask_np[0, 0]
    elif len(mask_np.shape) == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8) * 255

    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
    mask_filename = f"mask_{frame_index:03d}.png"
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask_resized)
    return mask_resized

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
        cv2.imwrite(frame_path, frame)

        mask_np = process_frame_with_sahi(
            frame_path=frame_path,
            sam_model_type=args.sam_model_type,
            detection_model_path="./checkpoints/yolov8x.pt",
            output_mask_dir=temp_masks_dir,
            frame_index=i
        )

        if mask_np.shape[:2] != frame.shape[:2]:
            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

        frames_list.append(frame)
        masks_list.append(mask_np)
        
    # Convert lists to numpy arrays and normalize
    frames_np = np.array(frames_list).astype(np.float32) / 255.0  # Shape: (len(frames), H, W, 3)
    masks_np = np.array(masks_list).astype(np.float32) / 255.0     # Shape: (len(masks), H, W)

    # Ensure mask_np is 2D per mask
    masks_np = masks_np.reshape(len(masks_list), masks_list[0].shape[0], masks_list[0].shape[1])

    print("frames: ", frames_np.shape)  # Expected shape: (len(frames), H, W, 3)
    print("masks: ", masks_np.shape)    # Expected shape: (len(masks), H, W)


    # Output directory for inpainted frames
    save_path = './output/inpainted_frames'
    os.makedirs(save_path, exist_ok=True)
    
    # Perform inpainting
    inpainted_frames = model.baseinpainter.inpaint(frames_np, masks_np, ratio=1)
    
    print("inpainted_frames_np: ", inpainted_frames.shape)
    
    # save
    for ti, inpainted_frame in enumerate(inpainted_frames):
        frame = Image.fromarray(inpainted_frame).convert('RGB')
        frame.save(os.path.join(save_path, f'{ti:03d}.png'))

    torch.cuda.empty_cache()
    print('switch to ori')

    print(f"Inpainted frames saved to {save_path}")

    # # Save as video
    # generate_video_from_frames(processed_frames, args.output_path, args.fps)
    # print(f"Output video saved to {args.output_path}")

if __name__ == "__main__":
    main()

# print("frames: ", frames_np.shape)
# print("masks: ", masks_np.shape)

# mask_np = process_frame_with_sahi(
#             frame_path=frame_path,
#             sam_model_type=args.sam_model_type,
#             detection_model_path="./checkpoints/yolov8x.pt",
#             output_mask_dir=temp_masks_dir,
#             frame_index=i
#         )

# # Add a channel dimension to mask_np and repeat it 3 times
#         frame_np = frame_np.reshape(720, 1280, 3, 1)
#         mask_np = mask_np[:, :, np.newaxis]
#         mask_np = np.repeat(mask_np, 3, axis=2)
        
#         print("frames: ", frame_np.shape)
#         print("masks: ", mask_np.shape)

#         # Perform inpainting using NumPy arrays
#         inpainted_frame_np = model.baseinpainter.inpaint(frame_np, mask_np, ratio=1)

#         # No need to convert back to tensor for saving
#         processed_frames.append((inpainted_frame_np * 255).astype(np.uint8))












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
    # cap = cv2.VideoCapture(video_path)
    # frames = []
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     else:
    #         break
    # cap.release()
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Determine the scaling factor, maintaining aspect ratio
                height, width = frame.shape[:2]
                max_dimension = max(height, width)
                scale = 640.0 / max_dimension if max_dimension > 640 else 1

                # Resize the frame, maintaining aspect ratio
                if scale != 1:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                # Convert color from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                # Check memory usage and stop if too high
                if psutil.virtual_memory().percent > 90:
                    print("Memory usage is too high (>90%). Stopping video extraction.")
                    break
            else:
                break
    except Exception as e:
        print(f"Error reading video source {video_path}: {e}")
    finally:
        # Always release the video capture
        cap.release()
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
    if frame is None:
        raise ValueError(f"Unable to load frame from {frame_path}")

    if isinstance(mask_image, torch.Tensor):
        mask_image_np = mask_image.cpu().numpy()
    else:
        mask_image_np = np.array(mask_image)
    
    print("ACSSAc: ", mask_image_np.shape)

    # Assuming the first dimension in mask_image_np corresponds to different detected objects
    if len(mask_image_np.shape) == 4 and mask_image_np.shape[0] > 1:
        # Select the second mask (index 1)
        mask_np = mask_image_np[0, 0]  # Modified index here
    elif len(mask_image_np.shape) == 3 and mask_image_np.shape[0] > 1:
        mask_np = mask_image_np[1]  # And here
        

    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8) * 255
    
    cv2.imwrite("DDDD.png", mask_np)

    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
    # mask_resized = mask_np.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)
    mask_filename = f"mask_{frame_index:03d}.png"
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask_resized)
    return mask_resized

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
        cv2.imwrite(frame_path, frame)

        mask_np = process_frame_with_sahi(
            frame_path=frame_path,
            sam_model_type=args.sam_model_type,
            detection_model_path="./checkpoints/yolov8x.pt",
            output_mask_dir=temp_masks_dir,
            frame_index=i
        )

        if mask_np.shape[:2] != frame.shape[:2]:
            mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

        frames_list.append(frame)
        masks_list.append(mask_np)
        
    frames_np = np.array(frames_list).astype(np.float32) / 255.0  # Shape: (len(frames), H, W, 3)
    masks_np = np.array(masks_list).astype(np.float32) / 255.0     # Shape: (len(masks), H, W)

    # Convert masks to binary (0 or 1)
    masks_np[masks_np > 0] = 1

    # Perform inpainting
    inpainted_frames = model.baseinpainter.inpaint(frames_np, masks_np, ratio=1)

    # Save inpainted frames
    save_path = './output/inpainted_frames'
    os.makedirs(save_path, exist_ok=True)

    for i, inpainted_frame in enumerate(inpainted_frames):
        inpainted_frame_uint8 = (inpainted_frame * 255).astype(np.uint8)
        Image.fromarray(inpainted_frame_uint8).save(os.path.join(save_path, f'inpainted_frame_{i:03d}.png'))

    generate_video_from_frames(inpainted_frames, args.output_path, args.fps)
    print(f"Inpainted frames saved to {save_path}")
    torch.cuda.empty_cache()
    print('switch to ori')


if __name__ == "__main__":
    main()
