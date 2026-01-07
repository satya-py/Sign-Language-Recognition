import os
import sys
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import uuid
import traceback

# Add parent directory to path to import model_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Configuration - Use absolute paths
MODEL_PATH = 'sign_language_model_best.h5'
ENCODER_PATH ='label_encoder.pkl'
VIDEO_DATA_DIR = 'videos_demo'

# Initialize Model
detector = None

def initialize_model():
    """Initialize the model with proper error handling"""
    global detector

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING SIGN LANGUAGE DETECTION SERVER")
    print("=" * 70)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Encoder Path: {ENCODER_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Encoder exists: {os.path.exists(ENCODER_PATH)}")
    print("=" * 70)
    
    try:
        # Import here to catch any import errors
        from model_loader import SignLanguageModel
        
        detector = SignLanguageModel(MODEL_PATH, ENCODER_PATH)
        
        print("\n" + "=" * 70)
        print("✅ SERVER INITIALIZATION SUCCESSFUL")
        print("=" * 70)
        print(f"Classes available: {detector.label_encoder.classes_.tolist()}")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ SERVER INITIALIZATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        traceback.print_exc()
        print("=" * 70 + "\n")
        return False

# Initialize on startup
initialize_model()

# In-memory storage for live sessions
live_sessions = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": detector is not None and detector.model is not None,
        "model_path": MODEL_PATH,
        "encoder_path": ENCODER_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "encoder_exists": os.path.exists(ENCODER_PATH)
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of available sign classes"""
    if not detector or not detector.label_encoder:
        return jsonify({"error": "Model not loaded"}), 500
    
    classes = detector.label_encoder.classes_.tolist()
    return jsonify({"classes": classes})

@app.route('/api/predict_video', methods=['POST'])
def predict_video():
    """Process uploaded video file - Uses model_loader.predict_video()"""
    print("\n" + "=" * 70)
    print("📹 VIDEO UPLOAD PREDICTION REQUEST")
    print("=" * 70)
    
    if not detector:
        print("❌ Model not initialized")
        return jsonify({"error": "Model not initialized"}), 500
    
    if 'file' not in request.files:
        print("❌ No file in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({"error": "No selected file"}), 400
    
    print(f"📁 Processing uploaded video: {file.filename}")
    
    # Save temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    print(f"💾 Saved to temp: {temp_path}")
    
    try:
        # Use the model's built-in predict_video method
        # This ensures consistency with preprocessing
        label, confidence = detector.predict_video(temp_path)
        
        print(f"\n🎯 PREDICTION:")
        print(f"   Label: {label}")
        print(f"   Confidence: {confidence:.2%}")
        print("=" * 70 + "\n")
        
        return jsonify({
            "result": label, 
            "confidence": float(confidence)
        })
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new live detection session"""
    if not detector:
        return jsonify({"error": "Model not initialized"}), 500
    
    session_id = str(uuid.uuid4())
    live_sessions[session_id] = {
        'sequence': [],
        'frame_count': 0,
        'last_prediction': None,
        'last_confidence': 0.0
    }
    print(f"\n🆕 New session started: {session_id}")
    return jsonify({"session_id": session_id})

@app.route('/api/session/end', methods=['POST'])
def end_session():
    """End a live detection session"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id and session_id in live_sessions:
        del live_sessions[session_id]
        print(f"🔚 Session ended: {session_id}")
        return jsonify({"status": "session ended"})
    
    return jsonify({"error": "Invalid session"}), 400

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Process single frame for real-time detection - Uses model_loader.extract_landmarks()"""
    if not detector:
        return jsonify({"error": "Model not initialized"}), 500
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400
    
    session_id = data.get('session_id')
    if not session_id or session_id not in live_sessions:
        return jsonify({"error": "Invalid or missing session_id. Call /api/session/start first"}), 400
    
    session = live_sessions[session_id]
    
    # Decode image
    try:
        # Handle data URL format: "data:image/jpeg;base64,..."
        image_str = data['image']
        if ',' in image_str:
            image_str = image_str.split(',')[1]
        
        image_data = base64.b64decode(image_str)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Extract landmarks using the model's method
    try:
        # Use model's extract_landmarks method - ensures consistency
        landmarks = detector.extract_landmarks(frame)
        session['sequence'].append(landmarks)
        session['frame_count'] += 1
        
        # Keep sliding window
        if len(session['sequence']) > detector.sequence_length:
            session['sequence'].pop(0)
        
        prediction = None
        confidence = 0.0
        
        # Make prediction when we have enough frames
        if len(session['sequence']) == detector.sequence_length:
            # Use model's predict_sequence method
            prediction, confidence = detector.predict_sequence(session['sequence'])
            session['last_prediction'] = prediction
            session['last_confidence'] = float(confidence)
            
            # Only log if confidence is high enough
            if confidence > 0.6:
                print(f"📊 Session {session_id[:8]}... - {prediction}: {confidence:.2%}")
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence) if confidence else 0.0,
            "frames_collected": len(session['sequence']),
            "frames_needed": detector.sequence_length,
            "ready": len(session['sequence']) == detector.sequence_length
        })
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/video_samples/<class_name>', methods=['GET'])
def get_video_sample(class_name):
    """Returns a video sample for the class"""
    # Try different filename formats
    possible_names = [
        f"{class_name.capitalize()}.mp4",
        f"{class_name}.mp4",
        f"{class_name.lower()}.mp4",
        f"{class_name.upper()}.mp4",
        f"{class_name.title()}.mp4"
    ]
    
    for filename in possible_names:
        file_path = os.path.join(VIDEO_DATA_DIR, filename)
        if os.path.exists(file_path):
            return send_from_directory(VIDEO_DATA_DIR, filename)
    
    return jsonify({
        "error": f"Video not found for {class_name}",
        "tried_names": possible_names,
        "video_dir": VIDEO_DATA_DIR
    }), 404

@app.route('/api/debug/session/<session_id>', methods=['GET'])
def debug_session(session_id):
    """Debug endpoint to check session state"""
    if session_id not in live_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = live_sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "frames_collected": len(session['sequence']),
        "total_frames_processed": session['frame_count'],
        "last_prediction": session['last_prediction'],
        "last_confidence": session['last_confidence'],
        "sequence_length_needed": detector.sequence_length if detector else None
    })

@app.route('/api/debug/sessions', methods=['GET'])
def debug_all_sessions():
    """Debug endpoint to list all active sessions"""
    return jsonify({
        "active_sessions": len(live_sessions),
        "sessions": {
            sid: {
                "frames": len(s['sequence']),
                "total_processed": s['frame_count'],
                "last_prediction": s['last_prediction']
            }
            for sid, s in live_sessions.items()
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 STARTING FLASK SERVER")
    print("=" * 70)
    
    if detector:
        print("✅ Model loaded successfully")
        print(f"📋 Available endpoints:")
        print(f"   - GET  /health")
        print(f"   - GET  /api/classes")
        print(f"   - POST /api/session/start")
        print(f"   - POST /api/predict_frame")
        print(f"   - POST /api/predict_video")
        print(f"   - POST /api/session/end")
        print(f"   - GET  /api/video_samples/<class_name>")
        print(f"\n🎯 Sign classes: {', '.join(detector.label_encoder.classes_.tolist())}")
    else:
        print("⚠️ WARNING: Model not loaded!")
        print("Server will start but predictions will fail")
    
    print(f"\n🌐 Server: http://localhost:5000")
    print(f"🏥 Health check: http://localhost:5000/health")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')