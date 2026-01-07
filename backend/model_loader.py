"""
Model loader for Sign Language Detection
Handles model loading, landmark extraction, and prediction
"""

import numpy as np
import cv2
import pickle
import os
import traceback

class SignLanguageModel:
    def __init__(self, model_path, encoder_path, sequence_length=30):
        """
        Initialize the sign language model
        
        Args:
            model_path: Path to the trained .h5 model
            encoder_path: Path to the label encoder .pkl file
            sequence_length: Number of frames in a sequence (default: 30)
        """
        print("\n" + "="*70)
        print("INITIALIZING SIGN LANGUAGE MODEL")
        print("="*70)
        
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = None
        self.mp_hands = None
        self.hands = None
        
        # Check file existence
        print(f"\n📁 Checking files...")
        print(f"   Model path: {model_path}")
        print(f"   Model exists: {os.path.exists(model_path)}")
        print(f"   Encoder path: {encoder_path}")
        print(f"   Encoder exists: {os.path.exists(encoder_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        # Load model
        try:
            print(f"\n🔄 Loading TensorFlow model...")
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            raise
        
        # Load encoder
        try:
            print(f"\n🔄 Loading label encoder...")
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Encoder loaded successfully")
            print(f"   Classes: {self.label_encoder.classes_.tolist()}")
            print(f"   Number of classes: {len(self.label_encoder.classes_)}")
        except Exception as e:
            print(f"❌ Failed to load encoder: {e}")
            traceback.print_exc()
            raise
        
        # Initialize MediaPipe
        try:
            print(f"\n🔄 Initializing MediaPipe...")
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✅ MediaPipe initialized")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            traceback.print_exc()
            raise
        
        print("\n" + "="*70)
        print("✅ MODEL INITIALIZATION COMPLETE")
        print("="*70 + "\n")
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a single frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            numpy array of shape (126,) containing landmarks for 2 hands
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # No hands detected - fill with zeros
                frame_landmarks = [0] * (21 * 3)
            
            # Pad or truncate to exactly 2 hands (126 features)
            if len(frame_landmarks) < 21 * 3 * 2:
                frame_landmarks.extend([0] * (21 * 3 * 2 - len(frame_landmarks)))
            else:
                frame_landmarks = frame_landmarks[:21 * 3 * 2]
            
            return np.array(frame_landmarks)
            
        except Exception as e:
            print(f"❌ Error extracting landmarks: {e}")
            # Return zeros if extraction fails
            return np.zeros(126)
    
    def predict_sequence(self, sequence):
        """
        Predict sign language gesture from a sequence of landmarks
        
        Args:
            sequence: List or array of landmark frames, shape (sequence_length, 126)
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        try:
            # Convert to numpy array
            sequence = np.array(sequence)
            
            # Validate shape
            if sequence.shape != (self.sequence_length, 126):
                print(f"⚠️ Warning: Expected shape ({self.sequence_length}, 126), got {sequence.shape}")
                
                # Try to fix the shape
                if len(sequence) < self.sequence_length:
                    # Pad with zeros
                    padding = np.zeros((self.sequence_length - len(sequence), 126))
                    sequence = np.vstack([sequence, padding])
                elif len(sequence) > self.sequence_length:
                    # Truncate
                    sequence = sequence[:self.sequence_length]
            
            # Add batch dimension: (1, sequence_length, 126)
            sequence = np.expand_dims(sequence, axis=0)
            
            # Make prediction
            predictions = self.model.predict(sequence, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Decode label
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return predicted_label, float(confidence)
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            traceback.print_exc()
            return "Error", 0.0
    
    def predict_video(self, video_path):
        """
        Predict sign language from a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        print(f"\n📹 Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        
        # Sample frames evenly
        if total_frames > self.sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        print(f"   Extracting {len(frame_indices)} frames...")
        
        sequence = []
        current_frame = 0
        
        for target_frame in frame_indices:
            # Skip to target frame
            while current_frame < target_frame:
                ret = cap.grab()
                if not ret:
                    break
                current_frame += 1
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            sequence.append(landmarks)
        
        cap.release()
        
        # Pad if needed
        while len(sequence) < self.sequence_length:
            sequence.append(np.zeros(126))
        
        # Make prediction
        label, confidence = self.predict_sequence(sequence[:self.sequence_length])
        
        print(f"   Result: {label} ({confidence:.2%})")
        
        return label, confidence
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.hands:
            self.hands.close()


# Test function
def test_model():
    """Test the model loader"""
    MODEL_PATH = r'C:\Users\satya\OneDrive\Desktop\projects\sign language\sign_language_model_best.h5'
    ENCODER_PATH = r'C:\Users\satya\OneDrive\Desktop\projects\sign language\label_encoder.pkl'
    VIDEO_PATH = r'C:\Users\satya\OneDrive\Desktop\projects\sign language\videos_demo\Cough.mp4'
    
    try:
        print("Testing model loader...")
        detector = SignLanguageModel(MODEL_PATH, ENCODER_PATH)
        print("\n✅ Model test successful!")
        
        # Test with dummy data
        print("\n🧪 Testing with dummy sequence...")
        dummy_sequence = np.random.rand(30, 126)
        label, conf = detector.predict_sequence(dummy_sequence)
        print(f"Dummy prediction: {label} ({conf:.2%})")
        print("\n🧪 Testing with video...")
        label, conf = detector.predict_video(VIDEO_PATH)
        print(f"Video prediction: {label} ({conf:.2%})")
        
        return True
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        traceback.print_exc()
        return False