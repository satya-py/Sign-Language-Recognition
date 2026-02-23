"""
Real-Time Sign Language Detection
Using your trained model for live webcam detection
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import sys
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RealTimeSignDetector:
    def __init__(self, model_path='sign_language_model_best.h5', 
                 encoder_path='label_encoder.pkl'):
        
        print("\n" + "="*70)
        print("REAL-TIME SIGN LANGUAGE DETECTION")
        print("="*70)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n[ERROR] Model file not found: {model_path}\n"
                f"   Please run train_sign_language_model.py first to train the model.\n"
            )
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"\n[ERROR] Label encoder not found: {encoder_path}\n"
                f"   Please run train_sign_language_model.py first to generate the label encoder.\n"
            )
        
        # Load model and encoder
        print(f"\n[INFO] Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model: {e}")
        
        print("[INFO] Loading label encoder...")
        try:
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load label encoder: {e}")
        
        print(f"\n[OK] Model loaded successfully!")
        print(f"[OK] Number of signs: {len(self.label_encoder.classes_)}")
        print(f"\n[INFO] Available signs:")
        for i, sign in enumerate(self.label_encoder.classes_, 1):
            print(f"   {i:2d}. {sign}")
        
        # Initialize MediaPipe
        print("\n[INFO] Initializing hand tracking...")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Settings
        self.sequence_length = 30
        self.sequence = []
        self.threshold = 0.6  # Confidence threshold (60%)
        
        print("[OK] Hand tracking initialized!")
        print("="*70 + "\n")
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        frame_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks = [0] * (21 * 3)
        
        # Pad for 2 hands (21 landmarks * 3 coordinates * 2 hands = 126 features)
        if len(frame_landmarks) < 21 * 3 * 2:
            frame_landmarks.extend([0] * (21 * 3 * 2 - len(frame_landmarks)))
        else:
            frame_landmarks = frame_landmarks[:21 * 3 * 2]
        
        return frame_landmarks, results
    
    def run(self, camera_index=0):
        """Run real-time detection"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_index}")
            print("[TIP] Try different camera index: 0, 1, or 2")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("[OK] Camera started!")
        print("\n" + "="*70)
        print("KEYBOARD CONTROLS")
        print("="*70)
        print("  'q' or 'ESC'  - Quit the application")
        print("  'r'           - Reset sequence (start fresh)")
        print("  'c'           - Clear current prediction")
        print("  SPACE         - Pause/Resume detection")
        print("\n[TIP] Perform signs slowly and clearly in front of camera")
        print("="*70 + "\n")
        
        current_prediction = None
        prediction_confidence = 0
        prediction_history = []  # Store recent predictions for smoothing
        paused = False
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                if not paused:
                    # Extract landmarks
                    landmarks, results = self.extract_landmarks(frame)
                    self.sequence.append(landmarks)
                    
                    # Keep only last sequence_length frames
                    if len(self.sequence) > self.sequence_length:
                        self.sequence = self.sequence[-self.sequence_length:]
                    
                    # Draw hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, 
                                hand_landmarks, 
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                    
                    # Make prediction when sequence is full
                    if len(self.sequence) == self.sequence_length:
                        try:
                            input_data = np.expand_dims(self.sequence, axis=0)
                            prediction = self.model.predict(input_data, verbose=0)
                            
                            predicted_class_idx = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class_idx]
                            
                            if confidence > self.threshold:
                                predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                                
                                # Add to history for smoothing
                                prediction_history.append(predicted_label)
                                if len(prediction_history) > 5:
                                    prediction_history.pop(0)
                                
                                # Use most common prediction in recent history
                                if len(prediction_history) >= 3:
                                    most_common = Counter(prediction_history).most_common(1)[0][0]
                                    current_prediction = most_common
                                    prediction_confidence = confidence
                                else:
                                    current_prediction = predicted_label
                                    prediction_confidence = confidence
                            else:
                                current_prediction = None
                        except Exception as e:
                            print(f"[WARN] Prediction error: {e}")
                
                # ============== DRAW UI ELEMENTS ==============
                
                # Semi-transparent overlay for top bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Frames counter
                frame_color = (0, 255, 0) if len(self.sequence) == self.sequence_length else (255, 165, 0)
                cv2.putText(frame, f"Frames: {len(self.sequence)}/{self.sequence_length}", 
                           (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, frame_color, 2)
                
                # Hand detection status
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    hand_text = f"[OK] {num_hands} Hand{'s' if num_hands > 1 else ''} Detected"
                    hand_color = (0, 255, 0)
                else:
                    hand_text = "[WARN] No Hands Detected"
                    hand_color = (0, 100, 255)
                
                cv2.putText(frame, hand_text, (w - 350, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
                
                # Pause indicator
                if paused:
                    cv2.putText(frame, "[PAUSED]", (w//2 - 100, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # Prediction display box
                if current_prediction and prediction_confidence > self.threshold:
                    # Calculate box size
                    box_y = 80
                    box_h = 140
                    box_margin = 20
                    
                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (box_margin, box_y), 
                                (w - box_margin, box_y + box_h), (0, 180, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # Draw border
                    cv2.rectangle(frame, (box_margin, box_y), 
                                (w - box_margin, box_y + box_h), (0, 255, 0), 4)
                    
                    # Sign name (larger text)
                    sign_text = current_prediction.upper()
                    cv2.putText(frame, sign_text, 
                               (box_margin + 20, box_y + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
                    
                    # Confidence bar
                    conf_text = f"Confidence: {prediction_confidence*100:.1f}%"
                    cv2.putText(frame, conf_text, 
                               (box_margin + 20, box_y + 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Draw confidence bar
                    bar_width = int((w - 2 * box_margin - 40) * prediction_confidence)
                    cv2.rectangle(frame, (box_margin + 20, box_y + 120), 
                                (box_margin + 20 + bar_width, box_y + 130), 
                                (255, 255, 255), -1)
                
                elif len(self.sequence) == self.sequence_length and not paused:
                    # Show "Low Confidence" message
                    cv2.putText(frame, "Low Confidence - Try Again", 
                               (w//2 - 200, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                
                # Bottom instruction bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 50), (w, h), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                instructions = "Q:Quit  |  R:Reset  |  C:Clear  |  SPACE:Pause"
                cv2.putText(frame, instructions, 
                           (w//2 - 300, h - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Sign Language Detection - Press Q to Quit', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[INFO] Quitting...")
                    break
                    
                elif key == ord('r'):
                    self.sequence = []
                    current_prediction = None
                    prediction_history = []
                    print("[INFO] Sequence reset")
                    
                elif key == ord('c'):
                    current_prediction = None
                    prediction_history = []
                    print("[INFO] Prediction cleared")
                    
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    status = "paused" if paused else "resumed"
                    print(f"[INFO] Detection {status}")
        
        except KeyboardInterrupt:
            print("\n\n[WARN] Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n[OK] Camera released")
            print("[OK] Detection stopped")
            print("="*70 + "\n")

# ==========================
# MAIN
# ==========================

def main():
    print("\n" + "="*70)
    print("  SIGN LANGUAGE DETECTION SYSTEM")
    print("="*70)
    
    try:
        # Initialize detector
        detector = RealTimeSignDetector(
            model_path='sign_language_model_best.h5',
            encoder_path='label_encoder.pkl'
        )
        
        # Run detection
        detector.run(camera_index=0)
        
    except FileNotFoundError as e:
        print(e)
        print("\n" + "="*70)
        print("[ERROR] FILE NOT FOUND")
        print("="*70)
        print("Make sure these files exist in your current directory:")
        print("  1. sign_language_model_best.h5")
        print("  2. label_encoder.pkl")
        print("\nCurrent directory:", os.getcwd())
        print("\n[TIP] Run train_sign_language_model.py first to generate these files")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*70)
        print("[TIP] TROUBLESHOOTING")
        print("="*70)
        print("1. Make sure packages are installed:")
        print("   pip install tensorflow opencv-python mediapipe")
        print("\n2. Check if your webcam is working")
        print("\n3. Try a different camera index (0, 1, or 2)")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()