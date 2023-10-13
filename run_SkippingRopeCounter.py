# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
import os.path as osp
import mediapipe as mp
import matplotlib.pyplot as plt

from utils import BufferList , FPS

from mediapipe_pose_estimator import PoseEstimator

    
class SkippingRopeCounter():
    def __init__(self , buffer_time=50 , point_std=0 , std_bias=0 , 
                 device="cpu" , draw=False) -> None:
        self.device = device
        self.draw = draw
        
        self.estimator = PoseEstimator(
            draw=self.draw , show_arm_angle=True , show_dis_line=True)
        self.point_std = 0
        self.std_bias = std_bias # 相对于标准值的偏差范围
        self.count = 0
        
        self._init_specs()
        print("[INFO] Initial Skipping Rope Counter Successfully")
    
    def _init_specs(self):
        self.fps = FPS().start()
        self.count = 0
        self.is_skipping = False
        
    def _frame_visualization(self , frame):
        return frame
    
    def _frame_analysis(self , srcImage):
        frame = self.estimator._inference_image(srcImage)
        frame = self._frame_visualization(frame)
        
        return frame
        
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help="video path")
    parser.add_argument('--save-dir', type=str, help="save dir")
    parser.add_argument('--draw', action='store_true', help="draw keypoints")
    parser.add_argument('--device', type=str, default='cuda', help="device.support ['cuda' , 'cpu']")
    parser.add_argument('--detection-confidence', type=float, default=0.5, help="min detection confidence")
    parser.add_argument('--tracking-confidence', type=float, default=0.5, help="min tracking confidence")
    
    args = parser.parse_args()
    
    # ------ Config Start ------ # 
    DRAW = args.draw
    DEVICE = args.device
    MIN_DETECTION_CONF = args.detection_confidence
    MIN_TRACKING_CONF = args.tracking_confidence
    # ------ Config End ------ # 
    video_path = args.video_path
    assert osp.exists(video_path) , \
        f"VideoFileNotFindError : {video_path} is not exist"
    cap = cv2.VideoCapture(video_path)
    
    save_path = args.save_dir
    if not osp.exists(save_path):
        os.mkdir(save_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(
    #     video_path.replace(".mp4", "_output.mp4"),
    #     fourcc , 20.0 , (int(cap.get(3)), int(cap.get(4))),
    # )
    
    skippingrope_counter = SkippingRopeCounter(
                device=DEVICE , draw=DRAW)
    
    while cap.isOpened():
        success , frame = cap.read()
        if not success:
            break
        
        time_start = time.time()
        result = skippingrope_counter._frame_analysis(frame)
        time_end = time.time()
        
        cv2.imshow("RealTime SkippingRopeCounter" , result)
        # out.write(result)
        if cv2.waitKey(1) & 0xFF in [ord('q') , 27]:
            break
    
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    return 
    
    
if __name__ == "__main__":
    main()

