# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
import os.path as osp
import mediapipe as mp
import matplotlib.pyplot as plt

from utils import BufferList

from mediapipe_pose_estimator import PoseEstimator

    
class SkippingRopeCounter():
    def __init__(self , buffer_time=50 , 
                 device="cpu" , show=False) -> None:
        self.estimator = PoseEstimator()
        
        self.std_bias = 0.25 # 相对于标准值的偏差范围
        
        self.buffer_time = buffer_time
        self.device = device
        self.show = show
        
        # For tracking counters and sharing states in and out of callbacks
        self.state_tracker = {
            "count" : 0
        }
        
        if self.show:
            self.buffer_time = 30
        
        self._reset_counter()
        
        print("[INFO] Initial Skipping Rope Counter Successfully")
        
    
    def _get_counter(self):
        return self.state_tracker["count"]
    
    def _reset_counter(self):
        self.state_tracker["count"] = 0
    
    def _preprocess(self , srcImage):
        image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        
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
        
        point = {
            "x" : center_hip_x , "y" : center_hip_y      
        }
        
        return point , shoulder_hip_y_distance
        
        
    def _plot(self , frame , results):
        self.mp_drawing.draw_landmarks(
            frame , results.pose_landmarks , self.mp_pose.POSE_CONNECTIONS , 
            landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()      
        )
        
        return frame
    
    def _inference_image(self , srcImage):
        self.image_height , self.image_width , _ = srcImage.shape
        # print(f"[INFO] frame width : {self.image_width} , frame height : {self.image_height}")
        frame = srcImage.copy()
        
        image = self._preprocess(srcImage)
        
        results = self._inference(image)
        
        # Indicates whether any detections are available or not.
        if results.pose_landmarks:
            self.mid_point , y_dis = self._postprocess(self.image_height , self.image_width , results)
            if self.mid_point["y"] > self.point_std + self.std_bias:
                if self.direct == 0:
                    self.state_tracker["count"] += 0.5
                    self.direct = 1
            if self.mid_point["y"] < self.point_std - self.std_bias:
                if self.direct == 1:
                    self.state_tracker["count"] += 0.5
                    self.direct = 0
                    
            # cv2.putText(frame, str(int(self._get_counter())), \
            #     (45, 460), cv2.FONT_HERSHEY_PLAIN, 7,(255, 0, 0), 8)
            
            cv2.putText( \
                frame , "Count : " + str(int(self._get_counter())), \
                (int(self.image_width * 0.6), int(self.image_height * 0.4)), \
                cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0, 255, 255) , 1, \
            )  
            
            if self.show:
                self._plot(frame , results)
                cv2.circle(frame, (self.mid_point["x"], self.mid_point["y"]), 5, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (self.mid_point["x"], self.mid_point["y"]), 10, (0, 0, 255), 2)
        else:
            # print("[INFO] No keypoints of the human body were detected")
            self.mid_point = {
                "x" : 0 , "y" : 0      
            }
            y_dis = 0
            
        # Flip the frame horizontally for a selfie-view display.
        
        return frame
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='xxx.mp4', help="video path")
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
        if cv2.waitKey(1) & 0xFF in [ord('q') , 27]:
            break
    
    skippingrope_counter._reset_counter()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 
    
    
if __name__ == "__main__":
    main()

