# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
import os.path as osp
import mediapipe as mp
import matplotlib.pyplot as plt

class BufferList():
    def __init__(self , buffer_time , default_time=0) -> None:
        self.buffer = [default_time for _ in range(buffer_time)]
    
    def push(self , value):
        self.buffer.pop(0)
        self.buffer.append(value)
    
    def max(self):
        return max(self.buffer)
    
    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0
        

class SkippingRopeCounter():
    def __init__(self , static_image_mode=False , min_detection_confidence=0.5 , 
                 min_tracking_confidence=0.5 , smooth_landmarks=True , buffer_time=50 , 
                 device="cpu" , show=False) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.static_image_mode = static_image_mode
        self.smooth_landmarks = smooth_landmarks # 平滑坐标防止抖动
        # The minimum confidence score for the pose detection to be considered successful.
        self.min_detection_conf = min_detection_confidence # 人体检测模型置信度阈值
        # The minimum confidence score for the pose tracking to be considered successful.
        self.min_tracking_conf = min_tracking_confidence # 姿态置信度阈值
        self.buffer_time = buffer_time
        self.device = device
        self.show = show
        
        # 标准点point_sd这个坐标是以视频开始第一帧画面是站在原地未起跳为前提
        self.point_std = 0
        self.direct = 0
        
        # left/right hip/shoulder point index      
        self.hip_landmarks_idx = [23 , 24] # hip
        self.shoulder_landmarks_idx = [11 , 12] # shoulder
        
        # Initialize and return Mediapipe FaceMesh Solution Graph object
        self.pose_model = self.mp_pose.Pose(
            static_image_mode = self.static_image_mode ,
            min_detection_confidence = self.min_detection_conf , 
            min_tracking_confidence = self.min_tracking_conf)

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "count" : 0
        }
        
        if self.show:
            self.buffer_time = 30
        
        self._reset_counter()
    
    def _get_counter(self):
        return self.state_tracker["count"]
    
    def _reset_counter(self):
        self.state_tracker["count"] = 0
    
    def _preprocess(self , srcImage):
        image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(srcImage)
        
        return image
    
    def _inference(self , image):
        results = self.pose_model.process(image)
        
        return results
    
    def _postprocess(self , frame_height , frame_width , results):
        hip_landmarks = [
            (landmark.x * frame_width , landmark.y * frame_height)
            for index , landmark in enumerate(results.pose_landmarks.landmark)
            if index in self.hip_landmarks_idx
        ]
        
        # 取左右盆骨点的中点作为盆骨判定点
        center_hip_x = int(np.mean([x[0] for x in hip_landmarks]))
        center_hip_y = int(np.mean([x[1] for x in hip_landmarks]))
        
        shoulder_landmarks = [
            (landmark.x * frame_width , landmark.y * frame_height)
            for index , landmark in enumerate(results.pose_landmarks.landmark)
            if index in self.shoulder_landmarks_idx
        ]
        
        center_shoulder_y = int(np.mean([x[1] for x in shoulder_landmarks]))
        
        shoulder_hip_y_distance = center_shoulder_y - center_hip_y
        
        return center_hip_x , center_hip_y
        
        
    def _plot(self , frame , results):
        self.mp_drawing.draw_landmarks(
            frame , results.pose_landmarks , self.mp_pose.POSE_CONNECTIONS , 
            landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()      
        )
        
        return frame
    
    def _inference_image(self , srcImage):
        self.image_height , self.image_width , _ = srcImage.shape
        frame = srcImage.copy()
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        srcImage.flags.writeable = False
        print(f"[INFO] frame width : {self.image_width} , frame height : {self.image_height}")
        
        image = self._preprocess(srcImage)
        
        results = self._inference(image)
        
        
        
        # Indicates whether any detections are available or not.
        if results.pose_landmarks:
            if self.show:
                self._plot(frame , results)
        else:
            print("[INFO] No keypoints of the human body were detected")
            center_hip_x = 0
            center_hip_y = 0
            cy_shoulder_hip = 0
            
        # Flip the frame horizontally for a selfie-view display.
        
        return image
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='xxx.mp4', help="video path")
    parser.add_argument('--save-dir', type=str, help="save dir")
    parser.add_argument('--draw', type=bool, default=False, help="draw keypoints")
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
    out = cv2.VideoWriter(
        video_path.replace(".mp4", "_output.mp4"),
        fourcc , 20.0 , (int(cap.get(3)), int(cap.get(4))),
    )
    
    skippingrope_counter = SkippingRopeCounter(device=DEVICE , show=DRAW)
    
    while cap.isOpened():
        success , frame = cap.read()
        if not success:
            break
        
        time_start = time.time()
        result = skippingrope_counter._inference_image(frame)
        time_end = time.time()
        fps = 1 / (time_end - time_start)
        
        cv2.imshow("RealTime SkippingRopeCounter" , result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    skippingrope_counter._reset_counter()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 
    
    
if __name__ == "__main__":
    main()

