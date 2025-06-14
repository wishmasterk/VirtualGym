# This is the main file --> updated
import os
import cv2
import time
import yaml
import numpy as np
import streamlit as st
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Set Streamlit page configuration
st.set_page_config(page_title="Virtual Gym", layout="wide")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# Load YOLO model
model_yolo = YOLO("yolov8n.pt")  # Using nano version for faster inference
# Load your trained model
model_path = 'EfficientNetB0.h5'
if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()
model_pose = load_model(model_path)

def load_poses():
    pose_info = {}
    poses_dir = "poses"
    
    for filename in os.listdir(poses_dir):
        if filename.endswith(('.yaml')):
            with open(os.path.join(poses_dir, filename), 'r') as f:
                pose_data = yaml.safe_load(f)
                pose_info[pose_data['name']] = pose_data
    return pose_info

def calculate_angle(a, b, c, epsilon=1e-6):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba) + epsilon
    norm_bc = np.linalg.norm(bc) + epsilon

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return 360 - angle if angle > 180 else angle

def get_landmark_coordinates(landmarks, landmark_idx):
    return [
        landmarks[landmark_idx].x,
        landmarks[landmark_idx].y,
        landmarks[landmark_idx].z
    ]

def contains_person(frame, confidence_threshold=0.3):
    """Check if frame contains exactly one person with confidence above threshold"""
    results = model_yolo(frame, verbose=False, classes=[0], imgsz=320)  # Only detect person (class 0)
    
    person_count = 0
    for result in results:
        for box in result.boxes:
            if box.conf > confidence_threshold:
                person_count += 1
    
    if person_count == 0:
        return (False, "⚠️ No person detected!")
    elif person_count == 1:
        return (True, None)
    else:
        return (False, "⚠️ Multiple people detected! Please ensure only one person is in frame.")

def create_progress_bar(current_step, total_steps):
    progress = current_step / total_steps
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    return f"Progress: |{bar}| {int(progress * 100)}%"

def create_feedback_display(frame, feedback_data, current_step):
    # Create a semi-transparent overlay for feedback
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate positions for feedback elements
    start_y = 50
    line_height = 30
    
    # Draw progress bar
    progress_bar = create_progress_bar(current_step, len(feedback_data))
    cv2.putText(overlay, progress_bar, (20, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw step instructions
    start_y += line_height * 2
    for step_num, step_data in feedback_data.items():
        # Choose colors based on step status
        if step_num < current_step:
            color = (0, 255, 0)  # Green for completed steps
            prefix = "✓"
        elif step_num == current_step:
            color = (0, 255, 255)  # Yellow for current step
            prefix = "►"
        else:
            color = (128, 128, 128)  # Gray for upcoming steps
            prefix = "○"
            
        text = f"{prefix} Step {step_num}: {step_data['instruction']}"
        cv2.putText(overlay, text, (20, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add feedback for current step if validation fails
        if step_num == current_step and 'current_feedback' in step_data:
            cv2.putText(overlay, f"   ⚠ {step_data['current_feedback']}", 
                       (40, start_y + line_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        start_y += line_height

    # Blend overlay with original frame
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def validate_pose_step(landmarks, step_data):
    """Validate current step based on angle requirements"""
    if not step_data.get('validation'):
        return True, None
        
    validation = step_data['validation']
    try:
        a = get_landmark_coordinates(landmarks, validation['landmarks'][0])
        b = get_landmark_coordinates(landmarks, validation['landmarks'][1])
        c = get_landmark_coordinates(landmarks, validation['landmarks'][2])
        
        angle = calculate_angle(a, b, c)
        min_val, max_val = validation['desired_range']
        
        if min_val <= angle <= max_val:
            return True, None
        return False, validation['feedback']
    except:
        return False, "Cannot detect required body parts"

def prepare_display_frame(frame, fps):
    display_frame = cv2.resize(frame, (640, 480))
    cv2.putText(display_frame, f"Frames/sec: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return display_frame

# Main page
def main_page():
    st.title("Virtual Yoga Instructor")
    
    # Add some styling
    st.markdown("""
        <style>
        .main-header {
            color: #2e86ab;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .pose-selector {
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .start-button {
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    selected_pose = st.selectbox("Choose Your Pose:", 
                               list(POSE_INFO.keys()),
                               key="pose-selector")

    # Display pose information before starting
    if selected_pose:
        pose_data = POSE_INFO[selected_pose]
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("About this Pose")
            st.write(pose_data["description"]["About"])
            
            st.subheader("Benefits")
            for benefit in pose_data["description"]["Benefits"]:
                st.write(f"• {benefit}")
                
        with col2:
            try:
                st.image(pose_data.get("sample_image", ""), 
                        caption="Pose Reference",
                        use_column_width=True)
            except:
                st.write("Sample image not available")

    if st.button("Start Practice", key="start-button"):
        st.session_state.current_pose = selected_pose
        st.rerun()

def pose_page():
    # Get pose information
    pose_info = POSE_INFO[st.session_state.current_pose]
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Initialize webcam frame placeholder
        frame_placeholder = st.empty()
    
    with col2:
        # Initialize feedback placeholder
        feedback_placeholder = st.empty()
        
        # Back button
        if st.button("← Back to Pose Selection"):
            del st.session_state.current_pose
            st.rerun()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize pose progression tracking
    current_step = 1
    progression_steps = pose_info['description']['Progression_Steps']
    step_completion_time = {}  # Track how long each step is maintained
    STEP_HOLD_TIME = 2.0  # Seconds to hold pose before progressing
    
    # FPS tracking variables
    fps_start_time = time.time()
    frames_this_second = 0
    current_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Update FPS count
        current_time = time.time()
        frames_this_second += 1
        
        if current_time - fps_start_time >= 1.0:
            current_fps = frames_this_second
            frames_this_second = 0
            fps_start_time = current_time

        # Process pose detection
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Validate current step
            step_data = progression_steps[current_step]
            is_valid, feedback = validate_pose_step(results.pose_landmarks.landmark, step_data)
            
            # Update step data with current feedback
            if feedback:
                step_data['current_feedback'] = feedback
            else:
                step_data.pop('current_feedback', None)
                
            # Track step completion
            if is_valid:
                if current_step not in step_completion_time:
                    step_completion_time[current_step] = current_time
                elif current_time - step_completion_time[current_step] >= STEP_HOLD_TIME:
                    if current_step < len(progression_steps):
                        current_step += 1
                        step_completion_time.clear()
            else:
                step_completion_time.clear()
            
            # Create feedback display
            frame = create_feedback_display(frame, progression_steps, current_step)
            
            # Add hold timer if step is being held correctly
            if current_step in step_completion_time:
                hold_time = current_time - step_completion_time[current_step]
                if hold_time < STEP_HOLD_TIME:
                    progress = int((hold_time / STEP_HOLD_TIME) * 100)
                    cv2.putText(frame, f"Hold: {progress}%", 
                              (frame.shape[1] - 150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add FPS counter
            frame = prepare_display_frame(frame, current_fps)

        # Display frame
        frame_placeholder.image(frame, channels="BGR")
        
        # If all steps completed, show success message
        if current_step > len(progression_steps):
            feedback_placeholder.success("🎉 Perfect pose achieved! Hold to maintain.")

    cap.release()

# Load pose information
POSE_INFO = load_poses()

# App routing
if "current_pose" not in st.session_state:
    main_page()
else:
    pose_page()