import cv2
import time
import mediapipe as mp

Holistic(
  static_image_mode=False, #specifies if input should be treated as static or video 
  model_complexity=1, #used to specify the complexity of the pose landmark model 
  smooth_landmarks=True, 
  min_detection_confidence=0.5, 
  min_tracking_confidence=0.5
)