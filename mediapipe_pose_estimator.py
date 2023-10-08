# -*- coding: utf-8 -*-

import numpy as np
import mediapipe as mp

class PoseEstimator():
    def __init__(self , min_detection_confidence=0.5 , 
                 min_tracking_confidence=0.5 , smooth_landmarks=True , 
                 angle_threshold=22) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.smooth_landmarks = smooth_landmarks # 平滑坐标防止抖动
        self.min_detection_conf = min_detection_confidence # 人体检测模型置信度阈值
        self.min_tracking_conf = min_tracking_confidence # 姿态置信度阈值
        self.ang_thresh = angle_threshold
    
        # left/right hip/shoulder/wrist point index      
        self.hip_landmarks_idx = [23 , 24] # hip
        self.shoulder_landmarks_idx = [11 , 12] # shoulder
        self.wrist_landmarks_idx = [15 , 16] # wrist
        
        # Initialize and return Mediapipe FaceMesh Solution Graph object
        self.pose_model = self.mp_pose.Pose(
            min_detection_confidence = self.min_detection_conf , 
            min_tracking_confidence = self.min_tracking_conf)
        
        print("[INFO] Initial People Pose Estimator Successfully")
        
        
    def _calculate_angle(self , a , b , c):
        """For calculating angle between three landmarks"""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return int(np.degrees(angle))

    def _is_skipping(self , lm):
        """calculate angles between hand, hips and shoulders"""
        left_angle = self._calculate_angle(lm[15], lm[11], lm[23])
        right_angle = self._calculate_angle(lm[16], lm[12], lm[24])
        avg_angle = (left_angle + right_angle) / 2
        result = True if (avg_angle > self.ang_thresh) else False 

        return result
    
    def _preprocess(self , ):
        pass
    
    def _inference(self , ):
        pass
    
    def _postprocess(self , ):
        pass
    
    def _inference_image(self , ):
        pass
    
