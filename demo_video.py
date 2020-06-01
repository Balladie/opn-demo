# -*- coding: utf-8 -*-

from __future__ import division
import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import os
import sys
import glob
import argparse


### My libs
sys.path.append('utils/')
sys.path.append('models/')
from utils.helpers import *
from models.OPN import OPN
from models.TCN import TCN

def get_arguments():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--input", type=str, default='parkour', required=True)
    parser.add_argument("--batch", type=int, default=10)
    return parser.parse_args()
args = get_arguments()
seq_name = args.input

save_path = ''


#################### Load Model
model = nn.DataParallel(OPN())
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
model.eval() 

pp_model = nn.DataParallel(TCN())
if torch.cuda.is_available():
    pp_model.cuda()
pp_model.load_state_dict(torch.load(os.path.join('TCN.pth')), strict=False)
pp_model.eval() 


#################### Load video
T = len(glob.glob(os.path.join('Video_inputs', seq_name, '*.jpg')))
tmp = cv2.imread(os.path.join('Video_inputs', seq_name, '00000.jpg'))
H, W = tmp.shape[0], tmp.shape[1]

for j in range(T // args.batch + 1):
    batch_size = args.batch

    frames = np.empty((args.batch, H, W, 3), dtype=np.float32)
    holes = np.empty((args.batch, H, W, 1), dtype=np.float32)
    dists = np.empty((args.batch, H, W, 1), dtype=np.float32)

    for i in range(args.batch):
        if i >= T:
            batch_size = i
            tmp_frames = frames
            tmp_holes = holes
            tmp_dists = dists
            frames = np.empty((batch_size, H, W, 3), dtype=np.float32)
            holes = np.empty((batch_size, H, W, 1), dtype=np.float32)
            dists = np.empty((batch_size, H, W, 1), dtype=np.float32)
            for k in range(batch_size):
                frames[k] = tmp_frames[k]
                holes[k] = tmp_holes[k]
                dists[k] = tmp_dists[k]
            break
        #### rgb
        img_file = os.path.join('Video_inputs', seq_name, '{:05d}.jpg'.format(i+j*batch_size))
        raw_frame = np.array(Image.open(img_file).convert('RGB'))/255.
        raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        frames[i] = raw_frame
        #### mask
        mask_file = os.path.join('Video_inputs', seq_name, '{:05d}.png'.format(i+j*batch_size))
        raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        raw_mask = (raw_mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
        holes[i,:,:,0] = raw_mask.astype(np.float32)
        #### dist
        dists[i,:,:,0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)

    frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
    holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
    dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
    # remove hole
    frames = frames * (1-holes) + holes*torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    # valids area
    valids = 1-holes
    # unsqueeze to batch 1
    frames = frames.unsqueeze(0)
    holes = holes.unsqueeze(0)
    dists = dists.unsqueeze(0)
    valids = valids.unsqueeze(0)


    ################### Inference
    MEM_EVERY = 5 # every 5 frame as memory frames
    comps = torch.zeros_like(frames)
    ppeds = torch.zeros_like(frames)

    # memory encoding 
    midx = list( range(0, batch_size, MEM_EVERY) )
    with torch.no_grad():
        mkey, mval, mhol = model(frames[:,:,midx], valids[:,:,midx], dists[:,:,midx])

    for f in range(batch_size):
        # memory selection
        if f in midx:
            ridx = [k for k in range(len(midx)) if k != int(f/MEM_EVERY)]
        else:
            ridx = list(range(len(midx)))

        fkey, fval, fhol = mkey[:,:,ridx], mval[:,:,ridx], mhol[:,:,ridx]
        # inpainting..
        for r in range(999): 
            if r == 0:
                comp = frames[:,:,f]
                dist = dists[:,:,f]
            with torch.no_grad(): 
                comp, dist = model(fkey, fval, fhol, comp, valids[:,:,f], dist)
        
            # update
            comp, dist = comp.detach(), dist.detach()
            if torch.sum(dist).item() == 0:
                break
        
        comps[:,:,f] = comp

    # post-processing...
    ppeds[:,:,0] = comps[:,:,0]
    hidden = None
    for f in range(batch_size):
        with torch.no_grad():
            pped,  hidden =\
                    pp_model(ppeds[:,:,f-1], holes[:,:,f-1], comps[:,:,f], holes[:,:,f], hidden)
            ppeds[:,:,f] = pped

    for f in range(batch_size):
        # visualize..
        est = (ppeds[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)
        #true = (frames[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8) # h,w,3
        #mask = (dists[0,0,f].detach().cpu().numpy() > 0).astype(np.uint8) # h,w,1
        #ov_true = overlay_davis(true, mask, colors=[[0,0,0],[0,100,100]], cscale=2, alpha=0.4)

        #canvas = np.concatenate([ov_true, est], axis=0)

        save_path = os.path.join('Video_results', seq_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #canvas = Image.fromarray(canvas)
        canvas = Image.fromarray(est)
        canvas.save(os.path.join(save_path, '{:05d}.jpg'.format(f+j*args.batch)))

print('Results are saved: ./{}'.format(save_path))
