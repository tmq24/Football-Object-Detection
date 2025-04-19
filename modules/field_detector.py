import cv2
import numpy as np
from typing import List, Tuple, Optional
from .model_provider import ModelProvider

class FieldDetector:
    def __init__(self):
        self.model = ModelProvider.get_player_detector()
        self.model.conf = 0.3
        
        self.field_hsv_ranges = [
            (np.array([30, 20, 20]), np.array([90, 255, 255])),
        ]
        
        self.morph_kernel = np.ones((7,7), np.uint8)
        
        self.line_params = dict(
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=50,
            maxLineGap=20
        )
        
    def create_players_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        mask for players, referees and staff
        """
        results = self.model(frame, verbose=False)
        
        players_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if not hasattr(results[0], 'boxes'):
            return players_mask
            
        for box in results[0].boxes:
            if int(box.cls[0]) in [0,3,4,5]:  
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                x1, y1 = max(0, x1-10), max(0, y1-10)
                x2, y2 = min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)
                cv2.rectangle(players_mask, (x1,y1), (x2,y2), 255, -1)
        
        players_mask = cv2.dilate(players_mask, self.morph_kernel, iterations=2)
        
        return players_mask
        
    def detect_field_mask(self, frame: np.ndarray, players_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """create mask for field area"""
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for low, high in self.field_hsv_ranges:
            mask |= cv2.inRange(hsv, low, high)
            
        if players_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(players_mask))
            
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
        
        mask = cv2.dilate(mask, self.morph_kernel, iterations=1)
        
        return mask
    
    def detect_field_lines(self, frame: np.ndarray, field_mask: np.ndarray, 
                          players_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """detect field lines"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 30, 150, apertureSize=3)
        
        edges = cv2.bitwise_and(edges, edges, mask=field_mask)
        if players_mask is not None:
            edges = cv2.bitwise_and(edges, cv2.bitwise_not(players_mask))
        
        lines = cv2.HoughLinesP(edges, **self.line_params)
        
        if lines is None:
            return []
            
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 50:  
                filtered_lines.append(line[0])
        
        return filtered_lines
    
    def classify_lines(self, lines: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """classify horizontal and vertical lines"""
        horizontals = []
        verticals = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 30 or angle > 150:
                horizontals.append(line)
            elif 60 < angle < 120:
                verticals.append(line)
                
        return horizontals, verticals
    
    def draw_debug(self, frame: np.ndarray, field_mask: Optional[np.ndarray] = None,
                  field_lines: Optional[List[np.ndarray]] = None,
                  players_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """draw debug info"""
        debug_frame = frame.copy()
        
        if players_mask is not None:
            mask_overlay = np.zeros_like(debug_frame)
            mask_overlay[players_mask > 0] = [0, 0, 255]  # Màu đỏ cho players mask
            debug_frame = cv2.addWeighted(debug_frame, 1, mask_overlay, 0.3, 0)
        
        if field_mask is not None:
            mask_overlay = np.zeros_like(debug_frame)
            mask_overlay[field_mask > 0] = [0, 255, 0]  # Màu xanh lá cho field mask
            debug_frame = cv2.addWeighted(debug_frame, 1, mask_overlay, 0.3, 0)
        
        if field_lines is not None:
            horizontals, verticals = self.classify_lines(field_lines)
            
            for line in horizontals:
                x1, y1, x2, y2 = line
                cv2.line(debug_frame, (x1,y1), (x2,y2), (0,255,255), 2)
                
            for line in verticals:
                x1, y1, x2, y2 = line
                cv2.line(debug_frame, (x1,y1), (x2,y2), (0,255,0), 2)
        
        return debug_frame
