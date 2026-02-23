"""
Diagnostic script to check model and encoder files
"""
import os
import pickle
import numpy as np

print("="*70)
print("MODEL DIAGNOSTICS")
print("="*70)

# Paths
MODEL_PATH = r'C:\Users\satya\OneDrive\Desktop\projects\sign language\sign_language_model_best.h5'
ENCODER_PATH = r'C:\Users\satya\OneDrive\Desktop\projects\sign language\label_encoder.pkl'

# Check file existence
print("\n1. FILE EXISTENCE CHECK")
print("-"*70)
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
print(f"Model path: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")

print(f"\nEncoder file exists: {os.path.exists(ENCODER_PATH)}")
print(f"Encoder path: {ENCODER_PATH}")
if os.path.exists(ENCODER_PATH):
    print(f"Encoder file size: {os.path.getsize(ENCODER_PATH)} bytes")

# Check encoder
print("\n2. ENCODER CHECK")
print("-"*70)
try:
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print(f"✅ Encoder loaded successfully")
    print(f"Type: {type(encoder)}")
    print(f"Classes: {encoder.classes_}")
    print(f"Number of classes: {len(encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading encoder: {e}")
    import traceback
    traceback.print_exc()

# Check TensorFlow/Keras
print("\n3. TENSORFLOW/KERAS CHECK")
print("-"*70)
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
except Exception as e:
    print(f"❌ Error importing TensorFlow: {e}")

# Try loading model
print("\n4. MODEL LOADING CHECK")
print("-"*70)
try:
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
    print(f"\nModel Summary:")
    model.summary()
    print(f"\nInput shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

# Check MediaPipe
print("\n5. MEDIAPIPE CHECK")
print("-"*70)
try:
    import mediapipe as mp
    print(f"✅ MediaPipe imported successfully")
    
    # Try initializing hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print(f"✅ MediaPipe Hands initialized")
except Exception as e:
    print(f"❌ Error with MediaPipe: {e}")
    import traceback
    traceback.print_exc()

# Check OpenCV
print("\n6. OPENCV CHECK")
print("-"*70)
try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ Error importing OpenCV: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70 + "\n")