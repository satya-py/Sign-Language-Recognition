"""
Sign Language Recognition Model Training Script
================================================

This script trains an LSTM-based model for real-time sign language recognition
from hand landmark sequences extracted from videos.

Dataset Structure:
    video_data/video_data/
        ├── class_1/
        │   ├── sample1.npy
        │   ├── sample2.npy
        │   └── ...
        ├── class_2/
        │   └── ...
        └── ...

Each .npy file contains:
    - Shape: (30, 126)
    - 30 frames per gesture
    - 126 features = 21 landmarks × 3 coordinates × 2 hands

Output:
    - sign_language_model.h5: Trained model
    - label_encoder.pkl: Label encoder for class mapping
"""

import numpy as np
import os
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# 1. DATASET LOADING AND VALIDATION
# ============================================================================

def load_dataset(data_path='video_data/video_data'):
    """
    Load and validate the sign language dataset from .npy files.
    
    Args:
        data_path: Path to the dataset directory containing class folders
        
    Returns:
        X: numpy array of shape (n_samples, 30, 126) - input sequences
        y: numpy array of shape (n_samples,) - class labels as strings
        label_encoder: Fitted LabelEncoder for converting labels to integers
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING AND VALIDATING DATASET")
    print("="*80)
    
    data_path = Path(data_path)
    
    # Check if path exists
    if not data_path.exists():
        # Try alternative path structure
        alt_path = Path('video_data')
        if alt_path.exists():
            print(f"[WARN] Path '{data_path}' not found. Using alternative: '{alt_path}'")
            data_path = alt_path
        else:
            raise FileNotFoundError(
                f"[ERROR] Dataset path not found: {data_path}\n"
                f"   Please ensure your dataset is in 'video_data/video_data/' or 'video_data/'"
            )
    
    X = []
    y = []
    invalid_files = []
    
    # Get all class folders (directories only)
    class_folders = sorted([d for d in data_path.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])
    
    if len(class_folders) == 0:
        raise ValueError(
            f"[ERROR] No class folders found in {data_path}\n"
            f"   Expected structure: {data_path}/class_name/*.npy"
        )
    
    print(f"\n[INFO] Found {len(class_folders)} sign language classes:")
    for i, folder in enumerate(class_folders, 1):
        print(f"   {i:2d}. {folder.name}")
    
    # Load data from each class folder
    print("\n[INFO] Loading data files...")
    for class_folder in class_folders:
        class_name = class_folder.name
        # Search recursively for .npy files (handles nested subdirectories)
        npy_files = list(class_folder.rglob('*.npy'))
        
        if len(npy_files) == 0:
            print(f"   [WARN] '{class_name}': No .npy files found, skipping...")
            continue
        
        loaded_count = 0
        for npy_file in npy_files:
            try:
                # Load the .npy file
                data = np.load(npy_file)
                
                # Handle 1D arrays (flattened data) - reshape to 2D
                if len(data.shape) == 1:
                    # Try to reshape: assume it's flattened (frames * features)
                    total_elements = data.shape[0]
                    target_frames = 30
                    target_features = 126
                    expected_elements = target_frames * target_features
                    
                    if total_elements == expected_elements:
                        # Perfect match - reshape directly
                        data = data.reshape(target_frames, target_features)
                    elif total_elements < expected_elements:
                        # Too few elements - pad with zeros
                        padding = np.zeros(expected_elements - total_elements, dtype=data.dtype)
                        data = np.concatenate([data, padding]).reshape(target_frames, target_features)
                    else:
                        # Too many elements - truncate
                        data = data[:expected_elements].reshape(target_frames, target_features)
                elif len(data.shape) == 2:
                    # Already 2D - process normally
                    pass
                else:
                    invalid_files.append((str(npy_file), f"Invalid dimensions: {data.shape}"))
                    continue
                
                # Ensure correct shape: (30, 126)
                target_frames = 30
                target_features = 126
                
                if len(data.shape) != 2:
                    invalid_files.append((str(npy_file), f"Could not convert to 2D: {data.shape}"))
                    continue
                
                current_frames, current_features = data.shape
                
                # Handle frame dimension
                if current_frames < target_frames:
                    # Pad with zeros if too short
                    padding = np.zeros((target_frames - current_frames, current_features))
                    data = np.vstack([data, padding])
                elif current_frames > target_frames:
                    # Truncate if too long (take first 30 frames)
                    data = data[:target_frames]
                
                # Handle feature dimension
                if current_features < target_features:
                    # Pad features with zeros
                    padding = np.zeros((target_frames, target_features - current_features))
                    data = np.hstack([data, padding])
                elif current_features > target_features:
                    # Truncate features (take first 126)
                    data = data[:, :target_features]
                
                # Final shape validation
                if data.shape != (target_frames, target_features):
                    invalid_files.append((str(npy_file), f"Shape mismatch after processing: {data.shape}"))
                    continue
                
                # Check for NaN or Inf values
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    invalid_files.append((str(npy_file), "Contains NaN or Inf values"))
                    continue
                
                X.append(data)
                y.append(class_name)
                loaded_count += 1
                
            except Exception as e:
                invalid_files.append((str(npy_file), str(e)))
        
        print(f"   [OK] '{class_name}': Loaded {loaded_count}/{len(npy_files)} files")
    
    # Check if we loaded any data
    if len(X) == 0:
        raise ValueError(
            "[ERROR] No valid data loaded!\n"
            "   Please check:\n"
            "   1. .npy files exist in class folders\n"
            "   2. Files contain valid numpy arrays\n"
            "   3. Arrays have compatible shapes"
        )
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Print invalid files if any
    if invalid_files:
        print(f"\n[WARN] {len(invalid_files)} files had issues:")
        for file_path, reason in invalid_files[:10]:  # Show first 10
            print(f"   - {Path(file_path).name}: {reason}")
        if len(invalid_files) > 10:
            print(f"   ... and {len(invalid_files) - 10} more")
    
    # Dataset summary
    print("\n" + "="*80)
    print("[INFO] DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(X)}")
    print(f"Input shape: {X.shape} (samples, frames, features)")
    print(f"Expected shape: (n, 30, 126)")
    print(f"\nSamples per class:")
    class_counts = Counter(y)
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name:25s}: {count:4d} samples")
    print("="*80)
    
    # Validate final shapes
    assert X.shape[1] == 30, f"Frame dimension mismatch: expected 30, got {X.shape[1]}"
    assert X.shape[2] == 126, f"Feature dimension mismatch: expected 126, got {X.shape[2]}"
    
    return X, y


# ============================================================================
# 2. LABEL ENCODING
# ============================================================================

def encode_labels(y):
    """
    Encode string labels to integers using LabelEncoder.
    
    Args:
        y: Array of string labels
        
    Returns:
        y_encoded: Array of integer labels
        label_encoder: Fitted LabelEncoder
    """
    print("\n" + "="*80)
    print("STEP 2: ENCODING LABELS")
    print("="*80)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n[OK] Encoded {len(label_encoder.classes_)} classes:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"   {i:2d} -> {class_name}")
    print("="*80)
    
    return y_encoded, label_encoder


# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================


# def biderctional_lstm_model(input_shape, num_classes):
#     """
#     Create an optimized LSTM-based model for sign language recognition.
    
#     Architecture:
#         - Input: (30, 126) - 30 frames × 126 features
#         - Bidirectional LSTM layers with dropout and batch normalization
#         - Dense layers for classification
#         - Output: num_classes (softmax)
    
#     Args:
#         input_shape: Tuple (30, 126) - frames and features
#         num_classes: Number of sign language classes
        
#     Returns:
#         Compiled Keras model
#     """
#     print("\n" + "="*80)
#     print("STEP 3: BUILDING MODEL ARCHITECTURE")
#     print("="*80)
    
#     model = keras.Sequential([
#         # Input layer
#         layers.Input(shape=input_shape, name='input_sequence'),
        
#         # Bidirectional LSTM layers
#         layers.Bidirectional(
#             layers.LSTM(
#                 units=128,
#                 return_sequences=True,
#                 activation='tanh',
#                 name='lstm_1'
#             ),
#             name='bidirectional_1'
#         ),
#         layers.Dropout(0.3, name='dropout_1'),
#         layers.BatchNormalization(name='bn_1'),
        
#         # Bidirectional LSTM layers
#         layers.Bidirectional(
#             layers.LSTM(
#                 units=256,
#                 return_sequences=True,
#                 activation='tanh',
#                 name='lstm_2'
#             ),
#             name='bidirectional_2'
#         ),
#         layers.Dropout(0.3, name='dropout_2'),
#         layers.BatchNormalization(name='bn_2'),
        
#         # Bidirectional LSTM layers
#         layers.Bidirectional(
#             layers.LSTM(
#                 units=512,
#                 return_sequences=False,
#                 activation='tanh',
#                 name='lstm_3'
#             ),
#             name='bidirectional_3'
#         ),
#         layers.Dropout(0.3, name='dropout_3'),
#         layers.BatchNormalization(name='bn_3'),
        
#         # Dense layers for classification
#         layers.Dense(
#             units=256,
#             activation='relu',
#             name='dense_1'
#         ),
#         layers.Dropout(0.3, name='dropout_4'),
#         layers.BatchNormalization(name='bn_4'),
        
#         # Output layer
#         layers.Dense(
#             units=num_classes,
#             activation='softmax',
#             name='output'
#         )
#     ])
    
#     # Compile the model
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     print("\n" + "="*80)
#     print("[OK] Model architecture created successfully.")
#     print("="*80)
    
#     return model

def create_lstm_model(input_shape, num_classes):
    """
    Create an optimized LSTM-based model for sign language recognition.
    
    Architecture:
        - Input: (30, 126) - 30 frames × 126 features
        - LSTM layers with dropout and batch normalization
        - Dense layers for classification
        - Output: num_classes (softmax)
    
    Args:
        input_shape: Tuple (30, 126) - frames and features
        num_classes: Number of sign language classes
        
    Returns:
        Compiled Keras model
    """
    print("\n" + "="*80)
    print("STEP 3: BUILDING MODEL ARCHITECTURE")
    print("="*80)
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input_sequence'),
        
        # First LSTM layer - captures short-term patterns
        layers.LSTM(
            units=128,
            return_sequences=True,
            activation='tanh',
            name='lstm_1'
        ),
        layers.Dropout(0.3, name='dropout_1'),
        layers.BatchNormalization(name='bn_1'),
        
        # Second LSTM layer - captures longer-term dependencies
        layers.LSTM(
            units=256,
            return_sequences=True,
            activation='tanh',
            name='lstm_2'
        ),
        layers.Dropout(0.3, name='dropout_2'),
        layers.BatchNormalization(name='bn_2'),
        
        # Third LSTM layer - final sequence processing
        layers.LSTM(
            units=128,
            return_sequences=False,  # Last LSTM, no sequences returned
            activation='tanh',
            name='lstm_3'
        ),
        layers.Dropout(0.3, name='dropout_3'),
        layers.BatchNormalization(name='bn_3'),
        
        # Dense layers for classification
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.4, name='dropout_4'),
        
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_5'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n[OK] Model created with input shape: {input_shape}")
    print(f"[OK] Number of classes: {num_classes}")
    print(f"[OK] Total parameters: {model.count_params():,}")
    print("\nModel Architecture:")
    print("-" * 80)
    model.summary()
    print("-" * 80)
    
    return model


# ============================================================================
# 4. TRAINING
# ============================================================================

def train_model(X, y_encoded, label_encoder, validation_split=0.2):
    """
    Train the LSTM model with early stopping and learning rate reduction.
    
    Args:
        X: Training data (n_samples, 30, 126)
        y_encoded: Encoded labels (n_samples,)
        label_encoder: LabelEncoder for saving
        validation_split: Fraction of data to use for validation
        
    Returns:
        Trained model and training history
    """
    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL")
    print("="*80)
    
    # Split data into train and validation sets (80/20)
    print(f"\n[INFO] Splitting dataset: {int((1-validation_split)*100)}% train, {int(validation_split*100)}% validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=validation_split,
        random_state=42,
        stratify=y_encoded  # Maintain class distribution
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Create model
    input_shape = (X.shape[1], X.shape[2])  # (30, 126)
    num_classes = len(label_encoder.classes_)
    model = create_lstm_model(input_shape, num_classes)
    
    # Define callbacks
    callbacks = [
        # Early stopping - stop if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Learning rate reduction - reduce LR when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpoint - save best model based on validation accuracy
        ModelCheckpoint(
            'sign_language_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train the model
    print("\n🚀 Starting training...")
    print("="*80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("="*80)
    
    return model, history


# ============================================================================
# 5. EVALUATION AND SAVING
# ============================================================================

def evaluate_and_save(model, X_val, y_val, label_encoder, history):
    """
    Evaluate the model and save it along with the label encoder.
    
    Args:
        model: Trained Keras model
        X_val: Validation data
        y_val: Validation labels
        label_encoder: LabelEncoder to save
        history: Training history
    """
    print("\n" + "="*80)
    print("STEP 5: EVALUATION AND SAVING")
    print("="*80)
    
    # Evaluate on validation set
    print("\n[INFO] Evaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Get training accuracy from history
    train_accuracy = max(history.history['accuracy'])
    final_train_accuracy = history.history['accuracy'][-1]
    
    print("\n" + "="*80)
    print("[INFO] TRAINING RESULTS")
    print("="*80)
    print(f"[OK] Training Accuracy:   {final_train_accuracy*100:.2f}% (best: {train_accuracy*100:.2f}%)")
    print(f"[OK] Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"[OK] Validation Loss:     {val_loss:.4f}")
    print("="*80)
    
    # Save model
    model_path = 'sign_language_model.h5'
    print(f"\n[INFO] Saving model to: {model_path}")
    model.save(model_path)
    print(f"   [OK] Model saved successfully")
    
    # Save label encoder
    encoder_path = 'label_encoder.pkl'
    print(f"[INFO] Saving label encoder to: {encoder_path}")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   [OK] Label encoder saved successfully")
    
    print("\n" + "="*80)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*80)
    print(f"\n[INFO] Saved files:")
    print(f"   - {model_path} (trained model)")
    print(f"   - sign_language_model_best.h5 (best model during training)")
    print(f"   - {encoder_path} (label encoder)")
    print(f"\n[INFO] Model is ready for real-time inference!")
    print("="*80 + "\n")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    try:
        print("\n" + "="*80)
        print("SIGN LANGUAGE RECOGNITION - MODEL TRAINING")
        print("="*80)
        print("Dataset: video_data/video_data/")
        print("Input shape: (30, 126) - 30 frames × 126 features")
        print("="*80)
        
        # Step 1: Load and validate dataset
        X, y = load_dataset('video_data/video_data')
        
        # Step 2: Encode labels
        y_encoded, label_encoder = encode_labels(y)
        
        # Step 3 & 4: Create and train model
        model, history = train_model(X, y_encoded, label_encoder, validation_split=0.2)
        
        # Step 5: Evaluate and save
        # Get validation set for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        evaluate_and_save(model, X_val, y_val, label_encoder, history)
        
        return model, label_encoder, history
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print("\n[TIP] Troubleshooting:")
        print("   1. Ensure dataset is in 'video_data/video_data/' or 'video_data/'")
        print("   2. Check that .npy files have compatible shapes")
        print("   3. Verify all required packages are installed:")
        print("      pip install tensorflow numpy scikit-learn")
        raise


if __name__ == "__main__":
    main()

