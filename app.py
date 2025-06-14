from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
import mediapipe as mp
import numpy as np
from pose_tracker import PoseTracker
from datetime import datetime, timedelta
import os

app = Flask(__name__, 
            static_url_path='',
            static_folder='static',
            template_folder='templates')

camera = None
pose_tracker = None
workout_history = []

def init_camera():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Could not open camera")
        return True
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        return False

def generate_frames():
    global camera, pose_tracker
    
    if not init_camera():
        return None
        
    if pose_tracker is None:
        pose_tracker = PoseTracker()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame with pose tracking
        frame = pose_tracker.process_frame(frame)
        
        # Convert frame to jpg format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not init_camera():
        return jsonify({"error": "Failed to initialize camera"}), 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_tracking')
def start_tracking():
    global camera, pose_tracker
    try:
        if not init_camera():
            return jsonify({"status": "error", "message": "Failed to initialize camera"}), 500
            
        exercise_type = request.args.get('exercise_type', 'bicep_curl')
        if pose_tracker is None:
            pose_tracker = PoseTracker()
        
        pose_tracker.start_tracking(exercise_type)
        return jsonify({"status": "success", "exercise_type": exercise_type})
    except Exception as e:
        print(f"Error starting tracking: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_tracking')
def stop_tracking():
    global pose_tracker, workout_history
    try:
        if pose_tracker:
            pose_tracker.stop_tracking()
            session_stats = pose_tracker.get_session_stats()
            if session_stats:
                workout_history.append({
                    'start_time': datetime.now().isoformat(),
                    'duration': session_stats.get('duration', 0),
                    'total_calories': session_stats.get('calories', 0),
                    'exercises': [{
                        'type': session_stats.get('exercise_type', 'unknown'),
                        'reps': session_stats.get('reps', 0),
                        'form_score': session_stats.get('form_score', 0)
                    }]
                })
            return jsonify({"status": "success", "stats": session_stats})
        return jsonify({"status": "error", "message": "No active tracking session"})
    except Exception as e:
        print(f"Error stopping tracking: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_stats')
def get_stats():
    global pose_tracker
    try:
        if pose_tracker:
            stats = pose_tracker.get_session_stats()
            return jsonify(stats if stats else {})
        return jsonify({})
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_workout_history')
def get_workout_history():
    global workout_history
    try:
        return jsonify({"status": "success", "history": workout_history})
    except Exception as e:
        print(f"Error getting workout history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/set_goals', methods=['POST'])
def set_goals():
    try:
        data = request.get_json()
        daily_calories = data.get('daily_calories', 300)
        weekly_workouts = data.get('weekly_workouts', 5)
        # Save goals (in a real app, this would go to a database)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error setting goals: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup')
def cleanup():
    global camera, pose_tracker
    try:
        if camera:
            camera.release()
            camera = None
        if pose_tracker:
            pose_tracker.stop_tracking()
            pose_tracker = None
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting AI Pose Estimation application...")
    try:
        print("Initializing Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
    finally:
        cleanup()
        print("Application shutdown complete.")