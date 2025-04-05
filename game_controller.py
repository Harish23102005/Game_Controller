import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Camera setup
cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None
last_gesture_time = 0
cooldown = 0.2  # Slightly increased to 200ms to prevent rapid repeats

# Track lane position (0 = left, 1 = center, 2 = right)
current_lane = 1  # Start in center

def detect_gesture(hand_landmarks, prev_x, prev_y):
    global last_gesture_time, current_lane
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    curr_x, curr_y = index_tip.x, index_tip.y
    
    if prev_x is None or prev_y is None:
        return None, curr_x, curr_y
    
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    
    vertical_threshold = 0.05
    horizontal_threshold = 0.03
    
    print(f"dx: {dx:.3f}, dy: {dy:.3f}, Lane: {current_lane}")
    
    current_time = time.time()
    if current_time - last_gesture_time < cooldown:
        return None, curr_x, curr_y
    
    if abs(dy) > vertical_threshold:
        if dy < 0:
            last_gesture_time = current_time
            return "jump", curr_x, curr_y
        else:
            last_gesture_time = current_time
            return "roll", curr_x, curr_y
    elif abs(dx) > horizontal_threshold:
        if dx < 0 and current_lane > 0:  # Move left if not already at left edge
            last_gesture_time = current_time
            current_lane -= 1
            return "left", curr_x, curr_y
        elif dx > 0 and current_lane < 2:  # Move right if not already at right edge
            last_gesture_time = current_time
            current_lane += 1
            return "right", curr_x, curr_y
    
    return None, curr_x, curr_y

def simulate_control(gesture):
    if gesture == "jump":
        pyautogui.press("up")
        print("Pressed: Up")
    elif gesture == "roll":
        pyautogui.press("down")
        print("Pressed: Down")
    elif gesture == "left":
        pyautogui.keyDown("left")
        time.sleep(0.05)  # Shortened to 50ms for a quick tap
        pyautogui.keyUp("left")
        print("Pressed: Left")
    elif gesture == "right":
        pyautogui.keyDown("right")
        time.sleep(0.05)
        pyautogui.keyUp("right")
        print("Pressed: Right")

# Focus the game window
pyautogui.click(100, 100)  # Adjust to your game window
time.sleep(1)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture, prev_x, prev_y = detect_gesture(hand_landmarks, prev_x, prev_y)
            if gesture:
                simulate_control(gesture)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Subway Surfers Finger Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()