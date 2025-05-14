# 🧼 Remove Selected Objects from Video using Tracking + Inpainting

This project demonstrates how to **automatically track and remove selected objects** from a video using a combination of:

- 🎯 YOLOv8 for object detection  
- 🧠 Segment Anything (SAM) for accurate object segmentation  
- 📽️ Object tracking for maintaining position across frames  
- 🧽 Deep learning-based **inpainting** (XMem + E2FGVI) to fill the removed region

---

## 🔍 Workflow Overview

1. The input video is loaded, and frames are extracted.
2. Objects are detected with YOLOv8 and segmented with SAM.
3. The object is tracked frame-by-frame.
4. A mask is generated per frame.
5. The object is removed via deep video inpainting.

---

## 🎥 Demo

### 🔹 Input & Tracking + Mask Generation

![combine](https://github.com/jonibek95/Tracking-Removing-Selected-objects/assets/84657258/0d913a5f-1526-47b6-b40e-19be58979e73)

---

### 🔹 Inpainted Result (Object Removed)

![inpainted_video](https://github.com/jonibek95/Tracking-Removing-Selected-objects/assets/84657258/9d85b582-9fd0-49b9-9f5c-8b765417da11)

---

## 📂 Output

- `./output/masks/`: Generated binary masks for each frame
- `./output/inpainted_frames/`: PNG images of inpainted frames
- `./output/inpainted.mp4`: Final inpainted video

---

## 🧠 Technologies Used

- `YOLOv8` — object detection
- `Segment Anything` — mask prediction
- `XMem` — object mask propagation
- `E2FGVI` — deep video inpainting
