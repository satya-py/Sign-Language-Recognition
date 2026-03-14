"""
Create Label Encoder from Dataset
==================================
This script creates the label_encoder.pkl file from the dataset structure.
"""

import pickle
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def create_label_encoder(data_path='video_data'):
    """
    Create and save label encoder from dataset class folders.
    
    Args:
        data_path: Path to dataset directory containing class folders
    """
    print("\n" + "="*70)
    print("CREATING LABEL ENCODER")
    print("="*70)
    
    data_path = Path(data_path)
    
    # Check if path exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset path not found: {data_path}\n"
            f"   Please ensure your dataset is in 'video_data/'"
        )
    
    # Get all class folders (directories only)
    class_folders = sorted([d for d in data_path.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])
    
    if len(class_folders) == 0:
        raise ValueError(
            f"[ERROR] No class folders found in {data_path}\n"
            f"   Expected structure: {data_path}/class_name/*.npy"
        )
    
    # Extract class names
    class_names = [folder.name for folder in class_folders]
    
    print(f"\n[INFO] Found {len(class_names)} sign language classes:")
    for i, class_name in enumerate(class_names, 1):
        print(f"   {i:2d}. {class_name}")
    
    # Create and fit label encoder
    print("\n[INFO] Creating label encoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    print(f"[OK] Encoded {len(label_encoder.classes_)} classes:")
    for i, class_name in enumerate(label_encoder.classes_, 1):
        encoded_value = label_encoder.transform([class_name])[0]
        print(f"   {i:2d}. {encoded_value} -> {class_name}")
    
    # Save label encoder
    encoder_path = 'label_encoder.pkl'
    print(f"\n[INFO] Saving label encoder to: {encoder_path}")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   [OK] Label encoder saved successfully")
    
    print("\n" + "="*70)
    print("[SUCCESS] LABEL ENCODER CREATED!")
    print("="*70)
    print(f"\n[INFO] Saved file: {encoder_path}")
    print(f"[INFO] You can now use this with detect_real_time.py")
    print("="*70 + "\n")
    
    return label_encoder

if __name__ == "__main__":
    try:
        create_label_encoder()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


