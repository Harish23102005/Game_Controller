import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller

class GestureGameController:
    def __init__(self):
        self.keyboard = Controller()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                        min_detection_confidence=0.7,
                                        min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=8, circle_radius=4)
        self.cap = cv2.VideoCapture(0)
        self.screen_width = 640
        self.screen_height = 480
        
        # Initialize position tracking variables
        self.last_x = None
        self.last_y = None
        self.movement_threshold = 0.02    # Reduced threshold for easier detection
        self.angle_threshold = 60         # Even wider angle for better detection
        self.min_movement_speed = 0.01    # Reduced minimum speed requirement
        self.gesture_locked = False
        self.current_gesture = None

    def detect_gesture(self, hand_landmarks):
        index_finger_x = hand_landmarks.landmark[8].x
        index_finger_y = hand_landmarks.landmark[8].y
        
        if self.last_x is None:
            self.last_x = index_finger_x
            self.last_y = index_finger_y
            return None
        
        # Calculate movement vector and speed
        dx = index_finger_x - self.last_x
        dy = index_finger_y - self.last_y
        movement_speed = np.sqrt(dx*dx + dy*dy)
        
        # Calculate movement angle
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Update position history
        self.last_x = index_finger_x
        self.last_y = index_finger_y
        
        # Only detect gestures for fast, deliberate movements
        if movement_speed < self.min_movement_speed:
            self.gesture_locked = False    # Allow new gestures more quickly
            return None
            
        # Reset gesture lock if movement is very small
        if movement_speed < self.min_movement_speed * 0.5:
            self.gesture_locked = False
            self.current_gesture = None
            return None
            
        # Check if gesture is locked
        if self.gesture_locked:
            if movement_speed > self.movement_threshold * 2:  # Allow new gesture if movement is strong
                self.gesture_locked = False
            else:
                return None
            
        # Detect only significant movements
        if movement_speed > self.movement_threshold:
            # First check for vertical movements with priority
            if abs(dy) > abs(dx) * 1.2:  # Vertical movement priority
                if dy < 0:  # Moving up
                    self.current_gesture = 'up'
                    self.gesture_locked = True
                else:  # Moving down
                    self.current_gesture = 'down'
                    self.gesture_locked = True
            # Only check horizontal if not moving vertically
            else:
                if abs(angle) < self.angle_threshold:  # Right
                    self.current_gesture = 'right'
                    self.gesture_locked = True
                elif abs(angle) > 180 - self.angle_threshold:  # Left
                    self.current_gesture = 'left'
                    self.gesture_locked = True
                
        return self.current_gesture

    def simulate_keypress(self, gesture):
        if not gesture:
            # Release all keys when no gesture
            for key in [Key.up, Key.down, Key.left, Key.right]:
                self.keyboard.release(key)
            return
            
        # Single keypress for each gesture
        if gesture == 'up':
            self.keyboard.press(Key.up)
            self.keyboard.release(Key.up)
        elif gesture == 'down':
            self.keyboard.press(Key.down)
            self.keyboard.release(Key.down)
        elif gesture == 'left':
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)
        elif gesture == 'right':
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)

    def run(self):
        while True:
            success, image = self.cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            
            # Draw guide lines
            h, w = image.shape[:2]
            cv2.line(image, (w//2, 0), (w//2, h), (128, 128, 128), 1)  # Vertical line
            cv2.line(image, (0, h//2), (w, h//2), (128, 128, 128), 1)  # Horizontal line
            
            # Process image and detect hands
            brightness_factor = 1.2
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=5)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw finger position only
                    index_finger_landmark = hand_landmarks.landmark[8]
                    h, w, c = image.shape
                    index_finger_point = (int(index_finger_landmark.x * w), int(index_finger_landmark.y * h))
                    cv2.circle(image, index_finger_point, 8, (0, 255, 0), -1)
                    
                    gesture = self.detect_gesture(hand_landmarks)
                    if gesture:
                        self.simulate_keypress(gesture)
                    else:
                        self.simulate_keypress(None)
            else:
                # Release all keys when hand is not detected
                self.simulate_keypress(None)

            # Display the image
            cv2.imshow('Game Controller', image)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureGameController()
    controller.run()