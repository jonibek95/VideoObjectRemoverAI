import PIL
from tqdm import tqdm

from PIL import Image
import os

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse



class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)
        self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint, args.device) 
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    def generator(self, images: list, template_mask:np.ndarray):
        
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i ==0:           
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
    
    
        
        
    
    
    