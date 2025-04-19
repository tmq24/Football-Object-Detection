import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time

class SpeedCalculator:
    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02)
        )
        
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.08,
            minDistance=8,
            blockSize=5
        )
        
        self.ransac_params = dict(
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=2000,
            confidence=0.99
        )
        
        self.camera_kalman = cv2.KalmanFilter(6, 6)
        self.camera_kalman.measurementMatrix = np.eye(6, dtype=np.float32)
        self.camera_kalman.transitionMatrix = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ], dtype=np.float32)
        self.camera_kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        self.motion_buffer = []
        self.motion_buffer_size = 5
        
        self.prev_gray = None
        self.prev_points = None
        self.prev_time = None
        self.prev_features = None
        self.tracks = defaultdict(list)
        self.speeds_history = defaultdict(list)
        
        self.M = None
        self.field_width = 105
        self.field_height = 68
        
        self.max_speed = 40
        self.min_speed = 0.5
        self.history_size = 5
        self.speed_threshold = 5
        
        self.debug = False

    def update_transform(self, field_lines: List[np.ndarray]) -> None:
        if not field_lines:
            if self.debug:
                print("No field lines found for transform")
            return
            
        horizontals = []
        verticals = []
        for line in field_lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 150:
                horizontals.append(line)
            elif 60 < angle < 120:
                verticals.append(line)
                
        if not horizontals or not verticals:
            if self.debug:
                print(f"Not enough lines for transform: {len(horizontals)} horizontal, {len(verticals)} vertical")
            return
            
        horizontals = sorted(horizontals, key=lambda l: l[1])
        verticals = sorted(verticals, key=lambda l: l[0])
        
        top_line = horizontals[0]
        bottom_line = horizontals[-1]
        
        left_line = verticals[0]
        right_line = verticals[-1]
        
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            px = ( (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4) ) / \
                 ( (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) )
            py = ( (x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4) ) / \
                 ( (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) )
            return int(px), int(py)
        
        try:
            top_left = line_intersection(top_line, left_line)
            top_right = line_intersection(top_line, right_line)
            bottom_left = line_intersection(bottom_line, left_line)
            bottom_right = line_intersection(bottom_line, right_line)
            
            if self.debug:
                print(f"Field corners: TL={top_left}, TR={top_right}, BL={bottom_left}, BR={bottom_right}")
            
            src_points = np.float32([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ])
            
            dst_points = np.float32([
                [0, 0],
                [self.field_width, 0],
                [self.field_width, self.field_height],
                [0, self.field_height]
            ])
            
            self.M = cv2.getPerspectiveTransform(src_points, dst_points)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating transform matrix: {str(e)}")
            self.M = None
        
    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.prev_time = None
        self.prev_features = None
        self.tracks.clear()
        self.speeds_history.clear()
        
    def get_average_speed(self, player_id: int, current_speed: float) -> float:
        if current_speed > self.max_speed:
            if len(self.speeds_history[player_id]) > 0:
                current_speed = min(current_speed, self.speeds_history[player_id][-1] * 1.5)
            else:
                return 0
                
        if len(self.speeds_history[player_id]) > 0:
            last_speed = self.speeds_history[player_id][-1]
            if abs(current_speed - last_speed) > self.speed_threshold:
                current_speed = last_speed * 0.7 + current_speed * 0.3
        
        self.speeds_history[player_id].append(current_speed)
        
        if len(self.speeds_history[player_id]) > self.history_size:
            self.speeds_history[player_id].pop(0)
            
        weights = np.exp(-0.5 * np.linspace(0, 2, len(self.speeds_history[player_id])) ** 2)
        weights = weights[::-1]
        weighted_speeds = np.array(self.speeds_history[player_id]) * weights
        avg_speed = np.sum(weighted_speeds) / np.sum(weights)
        
        return max(avg_speed, self.min_speed)
        
    def smooth_camera_motion(self, transform_matrix):
        dx = transform_matrix[0, 2]
        dy = transform_matrix[1, 2]
        da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        
        self.motion_buffer.append((dx, dy, da))
        if len(self.motion_buffer) > self.motion_buffer_size:
            self.motion_buffer.pop(0)
        
        weights = np.linspace(0.5, 1.0, len(self.motion_buffer))
        weights /= weights.sum()
        
        smooth_dx = 0
        smooth_dy = 0
        smooth_da = 0
        for (dx, dy, da), w in zip(self.motion_buffer, weights):
            smooth_dx += dx * w
            smooth_dy += dy * w
            smooth_da += da * w
        
        cos_a = np.cos(smooth_da)
        sin_a = np.sin(smooth_da)
        return np.array([
            [cos_a, -sin_a, smooth_dx],
            [sin_a, cos_a, smooth_dy]
        ], dtype=np.float32)

    def estimate_camera_motion(self, prev_gray, curr_gray):
        if self.prev_features is None:
            mask = np.ones_like(prev_gray)
            border = int(min(prev_gray.shape) * 0.1)
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
            
            self.prev_features = cv2.goodFeaturesToTrack(
                prev_gray, 
                mask=mask,
                **self.feature_params
            )
            return np.eye(2, 3)
            
        if self.prev_features is None or len(self.prev_features) < 4:
            return np.eye(2, 3)
            
        curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, 
            self.prev_features, None,
            **self.lk_params
        )
        
        if curr_features is None or len(curr_features) < 4:
            return np.eye(2, 3)
        
        prev_features_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray,
            curr_features, None,
            **self.lk_params
        )
        
        fb_error = np.abs(self.prev_features.reshape(-1, 2) - prev_features_back.reshape(-1, 2)).max(-1)
        good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 1.0)
        
        if not np.any(good):
            return np.eye(2, 3)
        
        good_old = self.prev_features[good].reshape(-1, 1, 2)
        good_new = curr_features[good].reshape(-1, 1, 2)
        
        if len(good_old) < 4:
            return np.eye(2, 3)
        
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, 
            **self.ransac_params
        )
        
        if transform_matrix is None:
            return np.eye(2, 3)
            
        measurement = np.array([
            transform_matrix[0,2],
            transform_matrix[1,2],
            np.arctan2(transform_matrix[1,0], transform_matrix[0,0]),
            0, 0, 0
        ], dtype=np.float32)
        
        self.camera_kalman.correct(measurement)
        prediction = self.camera_kalman.predict()
        
        dx, dy, da = prediction[:3]
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        predicted_matrix = np.array([
            [cos_a, -sin_a, dx],
            [sin_a, cos_a, dy]
        ], dtype=np.float32)
        
        smoothed_matrix = self.smooth_camera_motion(predicted_matrix)
        
        if len(good_new) < self.feature_params['maxCorners'] * 0.5:
            mask = np.ones_like(curr_gray)
            for x, y in good_new.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
            
            remaining_corners = self.feature_params['maxCorners'] - len(good_new)
            feature_params = self.feature_params.copy()
            feature_params['maxCorners'] = remaining_corners
            
            additional_features = cv2.goodFeaturesToTrack(
                curr_gray,
                mask=mask,
                **feature_params
            )
            
            if additional_features is not None:
                self.prev_features = np.vstack([good_new, additional_features])
            else:
                self.prev_features = good_new.reshape(-1, 1, 2)
        else:
            self.prev_features = good_new.reshape(-1, 1, 2)
            
        return smoothed_matrix
        
    def update(self, frame: np.ndarray, boxes: List[Tuple[int, float, Tuple[int, int, int, int]]],
              field_lines: Optional[List[np.ndarray]] = None) -> Dict[int, float]:
        if len(boxes) == 0:
            self.reset()
            return {}
            
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_time = time.time()
        
        if field_lines is not None:
            self.update_transform(field_lines)
            
        curr_points = []
        curr_ids = []
        for track_id, conf, (x1, y1, x2, y2) in boxes:
            if conf > 0.5:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                curr_points.append([center_x, center_y])
                curr_ids.append(track_id)
        
        if not curr_points:
            self.reset()
            return {}
            
        curr_points = np.array(curr_points, dtype=np.float32)
            
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = curr_gray
            self.prev_time = curr_time
            self.prev_points = curr_points
            return {}
            
        camera_motion = self.estimate_camera_motion(self.prev_gray, curr_gray)
        
        self.prev_points = self.prev_points.reshape(-1, 1, 2)
        curr_points = curr_points.reshape(-1, 1, 2)
        
        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, 
            self.prev_points, None, 
            **self.lk_params
        )
        
        speeds = {}
        dt = curr_time - self.prev_time
        if dt > 0 and self.M is not None:
            n_points = min(len(tracked_points), len(curr_ids))
            
            for i in range(n_points):
                if status[i][0] == 1:
                    try:
                        prev_point = self.prev_points[i][0]
                        curr_point = tracked_points[i][0]
                        
                        compensated_point = cv2.transform(
                            curr_point.reshape(1, 1, 2),
                            camera_motion
                        )[0][0]
                        
                        prev_real = cv2.perspectiveTransform(
                            prev_point.reshape(-1, 1, 2), self.M)[0][0]
                        curr_real = cv2.perspectiveTransform(
                            compensated_point.reshape(-1, 1, 2), self.M)[0][0]
                        
                        dx = curr_real[0] - prev_real[0]
                        dy = curr_real[1] - prev_real[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        speed = (distance / dt) * 3.6
                        
                        player_id = curr_ids[i]
                        avg_speed = self.get_average_speed(player_id, speed)
                        
                        if avg_speed > self.min_speed:
                            speeds[player_id] = avg_speed
                    except Exception as e:
                        continue
        
        self.prev_gray = curr_gray
        self.prev_time = curr_time
        self.prev_points = curr_points
        
        return speeds
        