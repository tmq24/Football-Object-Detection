import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from modules.speed_calculator import SpeedCalculator
from modules.field_detector import FieldDetector

def initialize_kalman_filter():
    """
    Initializes a Kalman Filter for tracking the ball.
    Returns:
        kf: KalmanFilter instance configured for ball tracking.
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
    kf.x = np.array([0., 0., 0., 0.])  # Initial state [x, y, vx, vy]
    
    # State transition matrix
    dt = 1.0  # time step
    kf.F = np.array([[1., 0., dt, 0.],
                     [0., 1., 0., dt],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    
    # Measurement function
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.]])
    
    # Measurement noise
    kf.R = np.array([[50., 0.],       # Adjust these values based on detection noise
                     [0., 50.]])
    
    # Process noise
    q = 100  # process noise
    kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                     [0, dt**4/4, 0, dt**3/2],
                     [dt**3/2, 0, dt**2, 0],
                     [0, dt**3/2, 0, dt**2]]) * q
    
    # Initial state covariance
    kf.P *= 1000.
    
    return kf

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels .

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    """
    Finds the images of the players in the frame and their bounding boxes.

    Args:
        result: ultralytics.engine.results.Results object that contains all the
        result of running the object detection algroithm on the frame

    Returns:
        players_imgs
            List of np.array objects that contain the BGR values of the cropped
            parts of the image that contains players.
        players_boxes
            List of ultralytics.engine.results.Boxes objects that contain various
            information about the bounding boxes of the players found in the image.
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1: y2, x1: x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Finds the kit colors of all the players in the current frame

    Args:
        players: List of np.array objects that contain the BGR values of the image
        portions that contain players.
        grass_hsv: tuple that contain the HSV color value of the grass color of
        the image background.

    Returns:
        kits_colors
            List of np arrays that contain the BGR values of the kits color of all
            the players in the current frame
    """
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([grass_hsv[0, 0, 0] - 15, 30, 30])
        upper_green = np.array([grass_hsv[0, 0, 0] + 15, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        height = player_img.shape[0]
        upper_third_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_third_mask[0:height//3] = 255
        
        combined_mask = cv2.bitwise_and(cv2.bitwise_not(grass_mask), upper_third_mask)
        
        blurred = cv2.GaussianBlur(player_img, (5, 5), 0)
        
        mean_color = cv2.mean(blurred, mask=combined_mask)
        kit_color = np.array(mean_color[:3])
        
        if not np.any(np.isnan(kit_color)):  
            kits_colors.append(kit_color)
            
    return kits_colors

def get_kits_classifier(kits_colors):
    """
    Creates a K-Means classifier that can classify the kits accroding to their BGR
    values into 2 different clusters each of them represents one of the teams

    Args:
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        kits_kmeans
            sklearn.cluster.KMeans object that can classify the players kits into
            2 teams according to their color..
    """
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors);
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit
    color

    Args:
        kits_classifer: sklearn.cluster.KMeans object that can classify the
        players kits into 2 teams according to their color.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        team
            np.array object containing a single integer that carries the player's
            team number (0 or 1)
    """
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf, frame_width):
    """
    Finds the label of the team that is on the left of the screen using multiple frames

    Args:
        players_boxes: List of ultralytics.engine.results.Boxes objects that
        contain various information about the bounding boxes of the players found
        in the image.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.
        kits_clf: sklearn.cluster.KMeans object that can classify the players kits
        into 2 teams according to their color.
    Returns:
        left_team_label
            Int that holds the number of the team that's on the left of the image
            either (0 or 1)
    """
    left_team_label = 0
    team_0_left = []  
    team_1_left = []  
    team_0_positions = []
    team_1_positions = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())
        center_x = (x1 + x2) / 2
        
        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        
        if team == 0:
            team_0_positions.append(center_x)
            if center_x < frame_width / 2:
                team_0_left.append(1)
        else:
            team_1_positions.append(center_x)
            if center_x < frame_width / 2:
                team_1_left.append(1)

    team_0_left_ratio = len(team_0_left) / len(team_0_positions) if team_0_positions else 0
    team_1_left_ratio = len(team_1_left) / len(team_1_positions) if team_1_positions else 0

    if team_0_left_ratio > 0.6:
        left_team_label = 0
    elif team_1_left_ratio > 0.6:
        left_team_label = 1
    else:
        team_0_avg = np.mean(team_0_positions) if team_0_positions else 0
        team_1_avg = np.mean(team_1_positions) if team_1_positions else 0
        left_team_label = 1 if team_0_avg > team_1_avg else 0

    return left_team_label


def annotate_video(video_path, model):
    """
    Process video with real-time tracking and annotation
    Args:
        video_path: Path to input video
        model: YOLOv8 model instance
    """
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = video_path.split('\\')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_out.mp4")
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    kits_clf = None
    left_team_label = None
    grass_hsv = None
    team_colors = {0: None, 1: None}  
    player_tracks = defaultdict(lambda: {"team": None, "color": None, "missed_frames": 0})
    ball_detections = []
    last_ball_pos = None
    
    # Initialize Kalman Filter for ball tracking
    ball_kf = initialize_kalman_filter()
    kalman_initialized = False
    
    # Initialize speed calculator and field detector
    speed_calculator = SpeedCalculator()
    field_detector = FieldDetector()
    
    # Colors for non-player objects
    other_colors = {
        2: (155, 62, 157),    # Ball - Purple
        3: (0, 255, 255),     # Main Referee - Yellow
        4: (217, 89, 204),    # Side Referee - Pink
        5: (22, 11, 15)       # Staff Member - Dark Gray
    }

    print("Processing video...")
    frame_count = 0
    print("\nModel Labels:", model.names)
    print("\nStarting video processing...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        annotated_frame = cv2.resize(frame, (width, height))

        # Field detection 
        players_mask = field_detector.create_players_mask(frame)
        field_mask = field_detector.detect_field_mask(frame, players_mask)
        field_lines = field_detector.detect_field_lines(frame, field_mask, players_mask)
        speed_calculator.update_transform(field_lines)
        
        # Run detection and tracking with ByteTrack
        results = model.track(
            annotated_frame,
            conf=0.3,
            verbose=False,
            tracker="bytetrack.yaml",
            persist=True,
            show=False
        )
        
        if not results:
            continue
            
        result = results[0]

        # Process first frame to get team colors
        if frame_count == 1:
            players_imgs, players_boxes = get_players_boxes(result)
            if players_imgs:  # Check if any players were detected
                kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)
                if len(kits_colors) >= 2:  # Need at least 2 players to classify teams
                    kits_clf = get_kits_classifier(kits_colors)
                    left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf, width)
                    grass_color = get_grass_color(result.orig_img)
                    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

                    # Get average team colors
                    teams = classify_kits(kits_clf, kits_colors)
                    for i, color in enumerate(kits_colors):
                        team = teams[i]
                        if team_colors[team] is None:
                            team_colors[team] = tuple(map(int, color))

        # Process detections
        if kits_clf is not None:  # Only process if team classification is initialized
            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                
                # Get tracking ID if available
                track_id = ""
                if hasattr(box, 'id'):
                    track_id = str(int(box.id.item()))
                
                # Get confidence score
                conf = float(box.conf.numpy()[0])
                
                # Use model's default label name
                label_name = model.names[label]
                label_text = f"{label_name}{track_id} ({conf:.2f})"
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count} - Detected: {label_name} (ID: {track_id}, Conf: {conf:.2f})")
                
                if label == 0:  # Player
                    if track_id is not None:
                        # Update or initialize player tracking
                        if player_tracks[track_id]["team"] is not None:
                            team = player_tracks[track_id]["team"]
                            box_color = player_tracks[track_id]["color"]
                            player_tracks[track_id]["missed_frames"] = 0
                        else:
                            # New player, determine team and color
                            player_img = result.orig_img[y1:y2, x1:x2]
                            kit_color = get_kits_colors([player_img], grass_hsv, annotated_frame)[0]
                            team = classify_kits(kits_clf, [kit_color])[0]
                            box_color = tuple(map(int, kit_color))
                            player_tracks[track_id]["team"] = team
                            player_tracks[track_id]["color"] = box_color
                            player_tracks[track_id]["missed_frames"] = 0
                            
                            # Update team colors if not set
                            if team_colors[team] is None:
                                team_colors[team] = box_color
                    
                    label_text = f"Player-{'L' if team == left_team_label else 'R'}{track_id}"

                elif label == 1:  # Goalkeeper
                    if x1 < width * 0.5:  # Left side GK
                        team = left_team_label
                        label_text = f"GK-L{track_id}"
                    else:  # Right side GK
                        team = 1 if left_team_label == 0 else 0
                        label_text = f"GK-R{track_id}"
                    box_color = team_colors[team] if team_colors[team] is not None else (128, 128, 128)

                elif label == 2:  # Ball
                    box_color = other_colors[2]
                    label_text = "Ball"
                    
                    ball_center_x = (x1 + x2) / 2
                    ball_center_y = (y1 + y2) / 2
                    
                    # Update Kalman Filter with measurement
                    if not kalman_initialized:
                        ball_kf.x = np.array([ball_center_x, ball_center_y, 0., 0.])
                        kalman_initialized = True
                    else:
                        ball_kf.predict()
                        ball_kf.update(np.array([ball_center_x, ball_center_y]))
                    
                    ball_detections.append({
                        'frame_idx': frame_count,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    })
                    last_ball_pos = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

                elif label == 3:  # Main referee
                    box_color = other_colors[3]
                    label_text = f"Main-Ref{track_id}"
                elif label == 4:  # Side referee
                    box_color = other_colors[4]
                    label_text = f"Side-Ref{track_id}"
                elif label == 5:  # Staff member
                    box_color = other_colors[5]
                    label_text = f"Staff{track_id}"
                else:  # Unknown object
                    box_color = (128, 128, 128)  # Gray for unknown
                    label_text = f"Unknown{track_id}"

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # If ball not detected but Kalman Filter is initialized, use prediction
        if not any(int(box.cls.numpy()[0]) == 2 for box in result.boxes) and kalman_initialized:
            ball_kf.predict()
            predicted_x, predicted_y = ball_kf.x[:2]
            
            # Convert predicted center to bounding box
            ball_size = 20  # Approximate ball size
            pred_x1 = int(predicted_x - ball_size/2)
            pred_y1 = int(predicted_y - ball_size/2)
            pred_x2 = int(predicted_x + ball_size/2)
            pred_y2 = int(predicted_y + ball_size/2)
            
            # Draw dashed rectangle for predicted ball position
            dash_length = 5
            for i in range(pred_x1, pred_x2, dash_length):
                cv2.line(annotated_frame, (i, pred_y1), (min(i + dash_length//2, pred_x2), pred_y1), 
                        other_colors[2], 2)
                cv2.line(annotated_frame, (i, pred_y2), (min(i + dash_length//2, pred_x2), pred_y2), 
                        other_colors[2], 2)
            
            for i in range(pred_y1, pred_y2, dash_length):
                cv2.line(annotated_frame, (pred_x1, i), (pred_x1, min(i + dash_length//2, pred_y2)), 
                        other_colors[2], 2)
                cv2.line(annotated_frame, (pred_x2, i), (pred_x2, min(i + dash_length//2, pred_y2)), 
                        other_colors[2], 2)
                
            cv2.putText(annotated_frame, "Ball", (pred_x1, pred_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, other_colors[2], 2)
        
        # Calculate and update speeds for tracked players
        boxes_for_speed = []
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Only calculate speed for players
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                track_id = int(box.id[0]) if box.id is not None else -1
                confidence = float(box.conf[0])
                boxes_for_speed.append((track_id, confidence, (x1, y1, x2, y2)))
        
        # Update speeds using SpeedCalculator
        speeds = speed_calculator.update(frame, boxes_for_speed, field_lines)
        
        # Display speeds
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Only display speed for players
                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id in speeds:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                    speed_text = f"{speeds[track_id]:.1f} km/h"
                    cv2.putText(annotated_frame, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        output_video.write(annotated_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    print("Completed!")
    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)

    model = YOLO(os.path.join(".", "weights", "last.pt"))
    video_path = os.path.normpath(sys.argv[1])
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    annotate_video(video_path, model)