"""
Preprocess a single video file to .npy format
Extracts hand landmarks and saves them for training
"""

import cv2
import numpy as np
import mediapipe as mp
import os

class VideoPreprocessor:
    def __init__(self, sequence_length=30):
        """Initialize MediaPipe for hand tracking"""
        print("\n" + "="*70)
        print("VIDEO PREPROCESSOR - Convert Video to Landmarks")
        print("="*70 + "\n")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_length = sequence_length
        
    def process_video(self, video_path):
        """Convert a video to landmark sequence"""
        print(f"📹 Processing video: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {total_frames/fps:.2f} seconds")
        
        landmarks_sequence = []
        
        # Calculate frame indices to extract (evenly distributed)
        if total_frames > self.sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        print(f"\n🔄 Extracting {len(frame_indices)} frames...")
        
        current_frame = 0
        processed_count = 0
        
        for target_frame in frame_indices:
            # Skip to target frame
            while current_frame < target_frame:
                ret = cap.grab()
                if not ret:
                    break
                current_frame += 1
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"   ⚠ Failed to read frame {target_frame}")
                break
            
            current_frame += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # If no hands detected, add zeros
                frame_landmarks = [0] * (21 * 3)
            
            # Pad for 2 hands (21 landmarks * 3 coords * 2 hands = 126 features)
            if len(frame_landmarks) < 21 * 3 * 2:
                frame_landmarks.extend([0] * (21 * 3 * 2 - len(frame_landmarks)))
            else:
                frame_landmarks = frame_landmarks[:21 * 3 * 2]
            
            landmarks_sequence.append(frame_landmarks)
            processed_count += 1
            
            # Progress indicator
            if processed_count % 5 == 0:
                print(f"   Progress: {processed_count}/{len(frame_indices)} frames")
        
        cap.release()
        
        # Pad sequence if too short
        while len(landmarks_sequence) < self.sequence_length:
            landmarks_sequence.append([0] * (21 * 3 * 2))
        
        if len(landmarks_sequence) < self.sequence_length:
            print(f"   ⚠ Padded {self.sequence_length - len(landmarks_sequence)} frames")
        
        result = np.array(landmarks_sequence[:self.sequence_length])
        
        print(f"\n✅ Processing complete!")
        print(f"   Output shape: {result.shape}")
        print(f"   Expected: ({self.sequence_length}, 126)")
        
        return result
    
    def save_landmarks(self, landmarks, output_path):
        """Save landmarks to .npy file"""
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created directory: {output_dir}")
        
        # Save as numpy array (np.save automatically adds .npy extension)
        np.save(output_path, landmarks)
        
        # Add .npy extension to check file size
        npy_path = output_path if output_path.endswith('.npy') else output_path + '.npy'
        print(f"💾 Saved to: {npy_path}")
        print(f"   File size: {os.path.getsize(npy_path)} bytes")

def preprocess_single_video(video_path, sign_name, output_base_dir=r'C:\Users\satya\OneDrive\Desktop\projects\sign language\video_data'):
    """
    Preprocess a single video and add it to the dataset
    
    Args:
        video_path: Path to the video file (e.g., 'allright.mp4')
        sign_name: Name of the sign (e.g., 'allright')
        output_base_dir: Base directory for video data
    """
    
    print("="*70)
    print("SINGLE VIDEO PREPROCESSING")
    print("="*70)
    print(f"Input video: {video_path}")
    print(f"Sign name: {sign_name}")
    print(f"Output directory: {output_base_dir}/{sign_name}")
    print("="*70 + "\n")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(sequence_length=30)
    
    # Process video
    landmarks = preprocessor.process_video(video_path)
    
    # Create output directory path
    output_dir = os.path.join(output_base_dir, sign_name)
    
    # Count existing files to get next number
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) 
                         if f.endswith('.npy')]
        next_number = len(existing_files)
    else:
        next_number = 0
    
    # Create output filename (number only, np.save will add .npy)
    output_filename = str(next_number)
    output_path = os.path.join(output_dir, output_filename)
    
    # Save landmarks
    preprocessor.save_landmarks(landmarks, output_path)
    
    print("\n" + "="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"✓ Video processed: {video_path}")
    print(f"✓ Saved as: {output_path}.npy")
    print(f"✓ Sign category: {sign_name}")
    print(f"✓ File number: {next_number}")
    print(f"\n📊 Current dataset for '{sign_name}':")
    
    # Show current count
    if os.path.exists(output_dir):
        total_files = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        print(f"   Total samples: {total_files}")
    
    print(f"\n📊 Next steps:")
    print(f"   1. Add more videos for '{sign_name}' if needed (recommended: 30+ samples)")
    print(f"   2. Run training: python train_model.py")
    print("="*70 + "\n")

def preprocess_multiple_videos():
    """
    Preprocess multiple videos at once
    Modify the video_list below with your videos
    """
    
    # List of videos to process: (video_path, sign_name)
    video_list = [
        (r"C:\Users\satya\OneDrive\Desktop\projects\sign language\allright.mp4", "allright"),
        # Add more videos here:
        # (r"path\to\video2.mp4", "sign_name2"),
        # (r"path\to\video3.mp4", "sign_name3"),
    ]
    
    print("\n" + "="*70)
    print("BATCH VIDEO PREPROCESSING")
    print("="*70)
    print(f"Total videos to process: {len(video_list)}\n")
    
    for idx, (video_path, sign_name) in enumerate(video_list, 1):
        print(f"\n{'='*70}")
        print(f"Processing video {idx}/{len(video_list)}")
        print(f"{'='*70}")
        
        try:
            preprocess_single_video(video_path, sign_name)
        except Exception as e:
            print(f"\n❌ Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("✅ BATCH PROCESSING COMPLETE!")
    print("="*70 + "\n")

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    import sys
    
    # You can either:
    # 1. Run with command line arguments
    # 2. Modify the code below directly
    
    # Method 1: Command line (uncomment to use)
    # if len(sys.argv) >= 3:
    #     video_path = sys.argv[1]
    #     sign_name = sys.argv[2]
    #     preprocess_single_video(video_path, sign_name)
    
    # Method 2: Direct specification (modify these lines)
    VIDEO_PATH = r"C:\Users\satya\OneDrive\Desktop\projects\sign language\allright.mp4"
    SIGN_NAME = "allright"
    
    try:
        preprocess_single_video(VIDEO_PATH, SIGN_NAME)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*70)
        print("💡 TROUBLESHOOTING")
        print("="*70)
        print("1. Check if video file exists:")
        print(f"   {VIDEO_PATH}")
        print("\n2. Make sure packages are installed:")
        print("   pip install opencv-python mediapipe numpy")
        print("\n3. Check file path format:")
        print("   - Use raw strings: r'C:\\path\\to\\file.mp4'")
        print("   - Or forward slashes: 'C:/path/to/file.mp4'")
        print("="*70 + "\n")