# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import mediapipe as mp

class PoseEstimator():
    def __init__(self , smooth_landmarks=True ,  min_detection_confidence=0.5 , 
                 min_tracking_confidence=0.5 , angle_threshold=18 , point_std=0 , 
                 std_bias=0 , draw=False , show_arm_angle=False , show_dis_line=False) -> None:
        self.results = None
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.smooth_landmarks = smooth_landmarks # 平滑坐标防止抖动
        self.min_detection_conf = min_detection_confidence # 人体检测模型置信度阈值
        self.min_tracking_conf = min_tracking_confidence # 姿态置信度阈值
        self.ang_thresh = angle_threshold
        self.point_std = point_std
        self.std_bias = std_bias
        
        self.draw = draw
        self.show_arm_angle = show_arm_angle
        self.show_dis_line = show_dis_line
    
        # left/right hip/shoulder/wrist point index      
        self.hip_landmarks_idx = [23 , 24] # hip
        self.shoulder_landmarks_idx = [11 , 12] # shoulder
        self.wrist_landmarks_idx = [15 , 16] # wrist
        self.ankle_landmarks_idx = [27 , 28]
        
        # Initialize and return Mediapipe FaceMesh Solution Graph object
        self.pose_model = self.mp_pose.Pose(
            smooth_landmarks = self.smooth_landmarks ,
            min_detection_confidence = self.min_detection_conf , 
            min_tracking_confidence = self.min_tracking_conf)
        
        print("[INFO] Initial People Pose Estimator Successfully")
        
        
    def _calculate_angle(self , a , b , c ,  direct=None):
        """For calculating angle between three landmarks"""
        ba = a - b 
        bc = c - b # 分别计算b->a,b->c的向量
        try:
            # 防止出现某个向量为[0 0]，导致计算的点积为0，出现在分母导致余弦值为NAN的异常
            ba_unique = np.unique(ba)
            bc_unique = np.unique(bc)
            
            if (ba_unique[0] == 0 and len(ba_unique) == 1) or \
                (np.unique(bc)[0] == 0 and len(np.unique(bc)) == 1):
                print("[INFO] Suspicious NAN cosine values appear")
                
                return 0 # 效果等价于 bc = ba
                
            # 使用向量的点积计算ba和bc之间的夹角的余弦值
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            # 反余弦计算夹角的弧度并转换为角度
            arccos = np.arccos(cosine_angle)
            # print(f"cosine_angle : {cosine_angle} , arccos : {arccos}")
    
            degrees = int(np.degrees(arccos))
        except Exception as e:
            print(f"[ERROR] {e}")

        return degrees

    def _get_landmarks(self , h , w):
        """Extract landmark x- and y-coordinates and return as int numyp array"""
        if self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            landmarks = [[lm.x, lm.y] for lm in landmarks]
            return np.multiply(landmarks, [w, h]).astype(int)
        else:
            return []
    
    def _get_avg_visibility(self):
        if self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            return np.mean([lm.visibility for lm in landmarks])
        else:
            return 0.0
    
    def _is_skipping(self):
        return 
    
    def _is_effective_angle(self , lm):
        left_angle = self._calculate_angle(lm[15], lm[11], lm[23] , direct="left")
        right_angle = self._calculate_angle(lm[16], lm[12], lm[24] , direct="right")
        avg_angle = (left_angle + right_angle) / 2

        if avg_angle > self.ang_thresh:
            return True
        else:
            return False
    
    def _preprocess(self , srcImage):
        image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _inference(self , image):
        results = self.pose_model.process(image)
        
        return results
    
    def _postprocess(self , frame_height , frame_width):
        hip_landmarks = [
            (landmark.x * frame_width , landmark.y * frame_height)
            for index , landmark in enumerate(self.results.pose_landmarks.landmark)
            if index in self.hip_landmarks_idx
        ]
        
        # 取左右盆骨点的中点作为盆骨判定点
        center_hip_x = int(
            np.mean([x[0] for x in hip_landmarks]))
        center_hip_y = int(
            np.mean([x[1] for x in hip_landmarks]))
        
        shoulder_landmarks = [
            (landmark.x * frame_width , landmark.y * frame_height)
            for index , landmark in enumerate(self.results.pose_landmarks.landmark)
            if index in self.shoulder_landmarks_idx
        ]
        
        center_shoulder_y = int(np.mean([x[1] for x in shoulder_landmarks]))
        
        shoulder_hip_y_distance = center_shoulder_y - center_hip_y
        
        point = {
            "x" : center_hip_x , "y" : center_hip_y      
        }
        
        return point , shoulder_hip_y_distance
    
    def _show_arm_angle(self , frame , lm):
        for i in [[16, 12, 24], [23, 11, 15]]:
            # get start angle and create new point
            new_lm = [lm[i[1]][0], lm[i[0]][1]] # 构建一个新的点计算角度
            direct = "left" if (i == 1) else "right"
            start_angle = self._calculate_angle(lm[i[0]], lm[i[1]], new_lm , direct) + 90  # grad
            angle = self._calculate_angle(lm[i[0]], lm[i[1]], lm[i[2]] , direct)
            end_angle = start_angle - angle

            cv2.ellipse(
                frame , center=(lm[i[1]][0], lm[i[1]][1]) , axes=(60, 60),
                angle=0 , startAngle=start_angle , endAngle=end_angle,
                color=(0, 255, 0) if self._is_effective_angle(lm) else (0, 0, 255),
                thickness=-1,
            )

            # print angle
            text_pos = np.mean([lm[j] for j in i], axis=0).astype(int)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str("{}".format(angle))
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(
                frame , text , (text_pos[0] - (textsize[0] // 2), text_pos[1]),
                font , 1 , (255, 255, 255) , 1 , 2,
            )

        # show lm
        lm_indices = [11, 12, 15, 16, 23, 24]
        for i in lm_indices:
            cv2.circle(frame, (lm[i][0], lm[i][1]), 8, (0, 0, 255), -1)

        # show lines
        lines = [[11, 23], [11, 15], [12, 24], [12, 16]]
        for l in lines:
            cv2.line(
                frame , (lm[l[0]][0], lm[l[0]][1]) , (lm[l[1]][0], lm[l[1]][1]),
                (0, 0, 255) , 3 ,
            )

        return frame
    
    def _plot(self , frame , results , lm):
        self.mp_drawing.draw_landmarks(
            frame , results.pose_landmarks , 
            self.mp_pose.POSE_CONNECTIONS , 
            landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()      
        )
        cv2.circle(
            frame, (self.mid_point["x"], self.mid_point["y"]), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(
            frame, (self.mid_point["x"], self.mid_point["y"]), 10, (0, 0, 255), 2)
        
        if self.show_arm_angle:
            frame = self._show_arm_angle(frame , lm)
        
        if self.show_dis_line:
            pass
        
        return frame
    
    def _inference_image(self , srcImage):
        image_height , image_width , _ = srcImage.shape
        frame = srcImage.copy()
        # print(f"[INFO] frame width : {self.image_width} , frame height : {self.image_height}")
        
        image = self._preprocess(srcImage)
        
        self.results = self._inference(image)
        
        landmarks = self._get_landmarks(image_height , image_width)
        # print("landmarks : " , landmarks)
        # Indicates whether any detections are available or not.

        if len(landmarks) != 0:
            self.mid_point , y_dis = self._postprocess(
                    image_height , image_width)
            # print("[INFO] Mid Point : " , self.mid_point)
            if self.draw:
                self._plot(frame , self.results , landmarks)
                
        else:
            # print("[INFO] No keypoints of the human body were detected")
            self.mid_point = {
                "x" : 0 , "y" : 0      
            }
            y_dis = 0
            
        # Flip the frame horizontally for a selfie-view display.
        
        return frame
    
