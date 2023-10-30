# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import mediapipe as mp

class PoseEstimator():
    def __init__(self , mode=None , smooth_landmarks=True ,  min_detection_confidence=0.5 , 
                 min_tracking_confidence=0.5 , angle_threshold=18 , point_std=10 , 
                 std_bias=5 , draw=False , show_arm_angle=False , show_dis_line=False , show_count=False) -> None:
        # Results
        self.results = None # mediapipe模型直接输出的结果
        self.landmarks = None # 经过图片宽高进行转换的二维坐标
        
        # Model
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Settings
        self.mode = mode
        self.smooth_landmarks = smooth_landmarks # 平滑坐标防止抖动
        self.min_detection_conf = min_detection_confidence # 人体检测模型置信度阈值
        self.min_tracking_conf = min_tracking_confidence # 姿态置信度阈值
        self.ang_thresh = angle_threshold
        
        self.feet_temp = 0
        self.std_bias = std_bias
        self.is_skipping = False
        self.is_effective_skipping = False
        self.position = 0 # 位姿，初始地面为0，正常跳绳判定线之间为1，超过判定线为2
        self.count_ = 0
        self.count = 0
        self.jump_anchor = 0 # 跳跃基准
        
        # Visualization
        self.draw = draw
        self.show_arm_angle = show_arm_angle
        self.show_jump_line = show_dis_line
        self.show_count = show_count
        self.shoulder_hip_y_distance = 0
    
        # left/right hip/shoulder/wrist point index      
        self.hip_landmarks_idx = [23 , 24] # hip
        self.shoulder_landmarks_idx = [11 , 12] # shoulder
        self.wrist_landmarks_idx = [15 , 16] # wrist
        self.ankle_landmarks_idx = [27 , 28] # ankle
        
        # Initialize and return Mediapipe FaceMesh Solution Graph object
        self.pose_model = self.mp_pose.Pose(
            smooth_landmarks = self.smooth_landmarks ,
            min_detection_confidence = self.min_detection_conf , 
            min_tracking_confidence = self.min_tracking_conf)
        
        print("[INFO] Initial People Pose Estimator Successfully")
        
        
    def _calculate_angle(self , a , b , c , direct=None):
        """For calculating angle between three landmarks"""
        ba = a - b 
        bc = c - b # 分别计算b->a,b->c的向量
        try:
            # 防止出现某个向量为[0 0]，导致计算的点积为0，出现在分母导致余弦值为NAN的异常
            ba_unique = np.unique(ba)
            bc_unique = np.unique(bc)
            
            if (ba_unique[0] == 0 and len(ba_unique) == 1) or \
                (np.unique(bc)[0] == 0 and len(bc_unique) == 1):
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
    
    def _get_count(self):
        return self.count
    
    def _is_effective_angle(self , lm):
        left_angle = self._calculate_angle(lm[15], lm[11], lm[23] , direct="left")
        right_angle = self._calculate_angle(lm[16], lm[12], lm[24] , direct="right")
        avg_angle = (left_angle + right_angle) / 2

        if avg_angle > self.ang_thresh:
            return True
        else:
            return False
    
    def _is_effective_jump(self , lm , height , width):
        """
            The skipping is counted by checking if the position of the feet of
        the user cross a certain height threshold. This threshold is calculated by
        considering the relative height of the user.
        """
        # adjust different jump heights
        if self.mode == "horizontal":
            distance_1_2 = int(height * 0.015)
            distance_1_3 = int(height * 0.093)
        elif self.mode == "vertical" :
            distance_1_2 = int(height * 0.010)
            distance_1_3 = int(height * 0.085)
        
        left_ankle , right_ankle =  [self.landmarks[x] for x in self.ankle_landmarks_idx]
        
        # 初始化跳跃基准高度为左脚踝和右脚踝中的较高值
        feet = left_ankle[1] if (left_ankle[1] > right_ankle[1]) else right_ankle[1] 
        anchor = feet
        
        # 计算出髋部和肩膀的Y坐标之差作为身体高度因子，通过插值调整，使其在特定范围内映射到另一个范围（0.2到0.4映射到0.8到1.2）
        height_factor = np.interp(self.shoulder_hip_y_distance, (0.2, 0.4), (0.8, 1.2))
        
        if self._get_avg_visibility() > 0.85:
            # if not self.is_skipping:
            if (not self.is_skipping):
                print(f"[INFO] Not in skipping state and initialize the anchor to {anchor}")
                self.jump_anchor = anchor # 如果不在跳跃状态，则将base设置anchor
                # 并调整distance距离
                distance_1_2 = int(distance_1_2 * height_factor)
                distance_1_3 = int(distance_1_3 * height_factor)
                print(f"[INFO] Not in skipping state and initialize the distance_1_2 to {distance_1_2} and distance_1_3 to {distance_1_3}")
                
            # Skipping-Counter mechanism
            line2 = int(self.jump_anchor - distance_1_2)
            line3 = int(self.jump_anchor - distance_1_3)

            if (not self.is_skipping):
                print(f"[INFO] Initialize skipping threshold bot line2 : {line2} and top line3 : {line3}")
            
            # 用以判断加速度方向,True为向下，False为向上
            down_direction = True if feet > self.feet_temp else False
            
            self.is_skipping = True
            # 首先要明确是，在OpenCV中，图像的左上角坐标点通常是 (0, 0)
            if feet < line2: 
                '''
                    若当前脚踝点高于下界且位姿为0时，判断为滞空状态
                '''
                if (self.position == 0) and (feet > line3):
                    if (not down_direction):
                        self.count += 1
                        self.position = 1
                        self.is_skipping = True
                    else:
                        # 若加速度向下，但仍高于下界，则是往后跳了
                        print(down_direction)
                        self.is_skipping = False 
                
            if feet > line2: 
                '''
                    若当前脚踝点低于下界，且状态为滞空，将位姿重新初始化为地面
                '''
                if self.position == 1:
                    self.position = 0
                    self.is_skipping = False
            if feet < line3:
                '''
                    若当前脚踝点高于上界，且为滞空状态，则判定为双摇状态，
                计数器加1，位姿设为2
                '''
                if self.position == 1: # first time over line 3
                    self.count += 1
                    self.position = 2
                    self.is_skipping = True
                    
            if anchor > line3: # double jump
                '''
                    若当前脚踝点低于上界，且位姿为2，则判定为双摇下落，
                ，将位姿设为2
                '''
                if self.position == 2:
                    self.position = 1
                    self.is_skipping = True
                    
            self.feet_temp = feet
            print(f"[DEBUG] Position : {self.position} , Count : {self.count}")
            return True
            
        else :
            # 骨骼关键点可见度不高
            print("[WARNING] Low visibility of key points of human skeleton")
            self.jump_anchor = 0
            self.is_skipping = False
            return False
    
    def _is_effective_skipping(self , height , width):
        is_effective_angle = self._is_effective_angle(self.landmarks)
        is_effective_jump = self._is_effective_jump(self.landmarks , height=height , width=width)
        
        if (is_effective_angle and is_effective_jump):
            return True
        else:
            return False
    
    def _preprocess(self , srcImage):
        image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _inference(self , image):
        results = self.pose_model.process(image)
        
        return results
    
    def _postprocess(self):
        hip_x_values = [self.landmarks[x][0] for x in self.hip_landmarks_idx]
        hip_y_values = [self.landmarks[x][1] for x in self.hip_landmarks_idx]
        center_hip_x = int(sum(hip_x_values) / len(hip_x_values))
        center_hip_y = int(sum(hip_y_values) / len(hip_y_values))
        # print(f"center_hip_x : {center_hip_x} , center_hip_y : {center_hip_y}")
        
        shoulder_x_values = [self.landmarks[x][0] for x in self.shoulder_landmarks_idx]
        shoulder_y_values = [self.landmarks[x][1] for x in self.shoulder_landmarks_idx]
        center_shoulder_x = int(sum(shoulder_x_values) / len(shoulder_x_values))
        center_shoulder_y = int(sum(shoulder_y_values) / len(shoulder_y_values))
        # print(f"center_shoulder_x : {center_shoulder_x} , center_shoulder_y : {center_shoulder_y}")
        
        shoulder_hip_y_distance = abs(center_shoulder_y - center_hip_y)
        # print(f"shoulder_hip_y_distance : {shoulder_hip_y_distance}")
        
        point = {
            "mid_hip_x" : center_hip_x , "mid_hip_y" : center_hip_y , 
            "mid_shoulder_x" : center_shoulder_x , "mid_shoulder_y" : center_shoulder_y ,       
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
    
    def _show_jump_line(self , frame , lm):
        height , width , _ = frame.shape
        if self.mode == "horizontal":
            distance_1_2 = int(height * 0.020)
            distance_1_3 = int(height * 0.097)
        else :
            distance_1_2 = int(width * 0.020)
            distance_1_3 = int(width * 0.097)
        
        left_ankle , right_ankle =  [self.landmarks[x] for x in self.ankle_landmarks_idx]
        line2 = int(self.jump_anchor - distance_1_2)
        line3 = int(self.jump_anchor - distance_1_3)
        
        cv2.line(frame , (0 , line2) , (width , line2) , (0 , 0 , 255) , 3) # bottom line , red
        cv2.line(frame , (0 , line3) , (width , line3) , (0 , 255 , 255) , 3) # top line , yellow
        
        feet_color = (
            [(0, 255, 0), (255, 255, 255)]
            if left_ankle[1] < right_ankle[1]
            else [(255, 255, 255), (0, 255, 0)]
        )
        cv2.circle(frame , (left_ankle[0] , left_ankle[1]) , 8 , feet_color[0] , -1)
        cv2.circle(frame , (right_ankle[0] , right_ankle[1]) , 8 , feet_color[1] , -1)
        
        return frame
    
    def _show_count(self , frame , count=None):
        x = 400
        y = 680
        cv2.rectangle(frame, (x - 20, y - 60), (x + 100, y + 20), (0, 0, 0), -1)
        count_text = count if count is not None else self.count
        cv2.putText(
            frame , str(count_text) , (x, y) , cv2.FONT_HERSHEY_SIMPLEX,
            2 , (255, 255, 255) , 3 , 2,)
        
        return frame
    
    def _plot(self , frame , results , lm):
        self.mp_drawing.draw_landmarks(
            frame , results.pose_landmarks , 
            self.mp_pose.POSE_CONNECTIONS , 
            landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()      
        )
        cv2.circle(
            frame, (self.mid_point["mid_hip_x"], self.mid_point["mid_hip_y"]), 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(
            frame, (self.mid_point["mid_hip_x"], self.mid_point["mid_hip_y"]), 10, (0, 0, 255), 2)
        
        if self.show_arm_angle:
            frame = self._show_arm_angle(frame , lm)
        
        if self.show_jump_line:
            frame = self._show_jump_line(frame , lm)
        
        if self.show_count:
            frame = self._show_count(frame)
        
        return frame
    
    
    
    def _inference_image(self , srcImage):
        image_height , image_width , _ = srcImage.shape
        frame = srcImage.copy()
        # print(f"[INFO] frame width : {self.image_width} , frame height : {self.image_height}")
        
        image = self._preprocess(srcImage)
        
        self.results = self._inference(image)
        
        self.landmarks = self._get_landmarks(image_height , image_width)
        # print("landmarks : " , landmarks)
        # Indicates whether any detections are available or not.
        if len(self.landmarks) != 0:
            self.mid_point , self.shoulder_hip_y_distance = self._postprocess()
            # print("[INFO] Mid Point : " , self.mid_point)
            self.is_effective_skipping = self._is_effective_skipping(height=image_height , width=image_width)
            if self.is_effective_skipping:
                self.count_ += 1
            if self.draw:
                self._plot(frame , self.results , self.landmarks)
        else:
            # print("[INFO] No keypoints of the human body were detected")
            self.mid_point = None
            self.shoulder_hip_y_distance = 0
            
        # Flip the frame horizontally for a selfie-view display.
        # frame = cv.flip(frame)
        
        return frame
    
