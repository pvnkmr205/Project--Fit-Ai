import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class PoseTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.is_tracking = False
        self.exercise_counter = 0
        self.exercise_type = "none"
        self.stage = "up"
        self.feedback = ""
        self.calories_burned = 0
        self.start_time = None
        
        # New attributes for enhanced features
        self.workout_history = []
        self.current_session = {
            'start_time': None,
            'exercises': [],
            'total_calories': 0,
            'duration': 0
        }
        self.rep_times = []  # Store timestamps for rep speed analysis
        self.form_score = 100  # Track form quality
        self.personal_goals = {
            'daily_calories': 300,
            'weekly_workouts': 5,
            'exercise_targets': {}
        }
        self.exercise_settings = {
            'bicep_curl': {'min_angle': 30, 'max_angle': 160, 'ideal_speed': 2.0},
            'squat': {'min_angle': 70, 'max_angle': 170, 'ideal_speed': 2.5},
            'pushup': {'min_angle': 80, 'max_angle': 160, 'ideal_speed': 2.0}
        }

    def start_tracking(self, exercise_type="bicep_curl"):
        self.is_tracking = True
        self.exercise_type = exercise_type
        self.exercise_counter = 0
        self.start_time = datetime.now()
        self.current_session = {
            'start_time': self.start_time,
            'exercises': [],
            'total_calories': 0,
            'duration': 0
        }
        self.rep_times = []
        self.form_score = 100

    def stop_tracking(self):
        if self.is_tracking:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            # Save session data
            self.current_session['duration'] = duration
            self.current_session['total_calories'] = self.calories_burned
            self.current_session['exercises'].append({
                'type': self.exercise_type,
                'reps': self.exercise_counter,
                'calories': self.calories_burned,
                'form_score': self.form_score
            })
            self.workout_history.append(self.current_session)
            
        self.is_tracking = False
        self.exercise_type = "none"
        self.start_time = None

    def analyze_form(self, angle, exercise_type):
        settings = self.exercise_settings[exercise_type]
        ideal_range = (settings['min_angle'], settings['max_angle'])
        
        # Deduct points for poor form
        if angle < ideal_range[0] - 10 or angle > ideal_range[1] + 10:
            self.form_score = max(0, self.form_score - 5)
            return "Poor form detected! Adjust your position."
        elif angle < ideal_range[0] - 5 or angle > ideal_range[1] + 5:
            self.form_score = max(0, self.form_score - 2)
            return "Form needs slight adjustment"
        return "Good form!"

    def analyze_rep_speed(self):
        if len(self.rep_times) < 2:
            return "Maintain steady pace"
            
        last_rep_time = self.rep_times[-1] - self.rep_times[-2]
        ideal_speed = self.exercise_settings[self.exercise_type]['ideal_speed']
        
        if last_rep_time < ideal_speed - 0.5:
            return "Slow down for better form"
        elif last_rep_time > ideal_speed + 0.5:
            return "Try to maintain a steady pace"
        return "Perfect speed!"

    def get_session_stats(self):
        if not self.current_session['start_time']:
            return None
            
        current_time = datetime.now()
        duration = (current_time - self.current_session['start_time']).total_seconds()
        
        return {
            'duration': duration,
            'reps': self.exercise_counter,
            'calories': self.calories_burned,
            'form_score': self.form_score,
            'exercise_type': self.exercise_type
        }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_calories(self, exercise_type, reps):
        # Approximate calories burned per rep
        calories_per_rep = {
            "bicep_curl": 0.32,
            "squat": 0.45,
            "pushup": 0.6
        }
        return calories_per_rep.get(exercise_type, 0.3) * reps

    def process_frame(self, frame):
        if not self.is_tracking:
            return frame

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Add stats overlay
                stats = self.get_session_stats()
                if stats:
                    cv2.putText(image, f"Exercise: {stats['exercise_type']}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Reps: {stats['reps']}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Calories: {stats['calories']:.1f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Form Score: {stats['form_score']}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if self.exercise_type == "bicep_curl":
                    # Get coordinates for bicep curl
                    shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    angle = self.calculate_angle(shoulder, elbow, wrist)
                    form_feedback = self.analyze_form(angle, self.exercise_type)
                    
                    # Count reps and analyze form
                    if angle > 160:
                        self.stage = "down"
                        self.feedback = f"{form_feedback} - Lower your arm slowly"
                    elif angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.exercise_counter += 1
                        self.rep_times.append(datetime.now().timestamp())
                        self.calories_burned = self.calculate_calories(self.exercise_type, self.exercise_counter)
                        speed_feedback = self.analyze_rep_speed()
                        self.feedback = f"{form_feedback} - {speed_feedback}"

                elif self.exercise_type == "squat":
                    # Get coordinates for squat
                    hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = self.calculate_angle(hip, knee, ankle)
                    
                    if angle < 100:
                        self.stage = "down"
                        self.feedback = "Good depth! Push through heels"
                    elif angle > 160 and self.stage == "down":
                        self.stage = "up"
                        self.exercise_counter += 1
                        self.calories_burned = self.calculate_calories(self.exercise_type, self.exercise_counter)
                        self.feedback = "Keep your back straight"

                elif self.exercise_type == "pushup":
                    # Get coordinates for pushup
                    shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    angle = self.calculate_angle(shoulder, elbow, wrist)
                    form_feedback = self.analyze_form(angle, self.exercise_type)
                    
                    if angle < 90:
                        self.stage = "down"
                        self.feedback = f"{form_feedback} - Keep core tight"
                    elif angle > 160 and self.stage == "down":
                        self.stage = "up"
                        self.exercise_counter += 1
                        self.rep_times.append(datetime.now().timestamp())
                        self.calories_burned = self.calculate_calories(self.exercise_type, self.exercise_counter)
                        speed_feedback = self.analyze_rep_speed()
                        self.feedback = f"{form_feedback} - {speed_feedback}"

                # Display stats on frame
                stats_color = (255, 255, 255)
                cv2.putText(image, f'Exercise: {self.exercise_type}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
                cv2.putText(image, f'Counter: {self.exercise_counter}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
                cv2.putText(image, f'Stage: {self.stage}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
                cv2.putText(image, f'Calories: {self.calories_burned:.1f}', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
                cv2.putText(image, f'Feedback: {self.feedback}', (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)

        except Exception as e:
            print(f"Error processing landmarks: {str(e)}")

        return image

    def get_stats(self):
        duration = 0
        if self.start_time:
            duration = (datetime.now() - self.start_time).seconds
        
        return {
            "exercise_type": self.exercise_type,
            "reps": self.exercise_counter,
            "calories": self.calories_burned,
            "duration": duration,
            "feedback": self.feedback
        }

    def __del__(self):
        self.pose.close()