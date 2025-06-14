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
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

# Load YOLO model
model_yolo = YOLO("yolov8m.pt")  # Using medium version for faster inference

# Load your trained model
model_path = 'MobileNetV3.keras'
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
                pose_data = yaml.safe_load(f)  # or json.load(f)
                pose_info[pose_data['name']] = pose_data
    return pose_info

# Angle Calculation
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
    
    # Count high-confidence person detections
    person_count = 0
    for result in results:
        for box in result.boxes:
            if box.conf > confidence_threshold:
                person_count += 1
    
    if person_count == 0:
        return (False, "⚠️ No person detected!")
    elif person_count == 1:
        return (True, None)  # No message needed for success case
    else:  # More than one person
        return (False, "⚠️ Multiple people detected! Please ensure only one person is in frame.")
    
def prepare_display_frame(frame , fps):
    display_frame = cv2.resize(frame, (640, 480))
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return display_frame
    
# Pose descriptions
POSE_INFO = load_poses()

# Main page
def main_page():
    st.title("Virtual Yoga Instructor")
    selected_pose = st.selectbox("Choose Your Pose:", list(POSE_INFO.keys()))

    if st.button("Start Practice"):
        st.session_state.current_pose = selected_pose
        st.rerun()


# Pose practice page
def pose_page():
    st.title(f"Practice: {st.session_state.current_pose}")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display pose description and sample image
        pose_info = POSE_INFO[st.session_state.current_pose]
        st.markdown(pose_info["description"])
        
        try:
            st.image(pose_info.get("sample_image", ""), 
                    caption="Proper Form Example",
                    width=300)
        except:
            st.warning("Sample image not available")
    
    with col2:
        if st.button("Back to Main Menu"):
            del st.session_state.current_pose
            st.rerun()

    # Initialize webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_placeholder = st.empty()
    feedback_placeholder = st.empty()
    target_class = POSE_INFO[st.session_state.current_pose]["class_index"]
    angle_configs = POSE_INFO[st.session_state.current_pose]["angles"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start FPS counter
        start_time = time.time()

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Person detection check
        person_valid, person_message = contains_person(frame)
        
        if not person_valid:
            fps = 1 / (time.time() - start_time)
            frame_placeholder.image(prepare_display_frame(frame, fps), channels="BGR")
            if person_message:  # Only show message if one exists
                feedback_placeholder.warning(person_message)
            continue

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feedback_messages = []
        # Set a confidence threshold (e.g., 0.5 for 50% confidence)
        confidence_threshold = 0.5

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:  # Iterate over all landmarks
                if landmark.visibility > confidence_threshold:
                    # Draw the pose landmarks on the frame
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Model prediction
            input_frame = cv2.resize(frame, (224, 224)) / 255.0
            prediction = model_pose.predict(np.expand_dims(input_frame, axis=0), verbose=0)
            confidence = np.max(prediction, axis=1)[0] * 100
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Check confidence threshold
            if confidence < 0.7:
                feedback_placeholder.warning("⚠ Low confidence: Not a recognized pose. Adjust your position.")
            else:
                # Only proceed with angle checks if confidence is high
                if predicted_class == target_class:
                    feedback_messages.append(f"✅ Correct pose! ({confidence * 100:.1f}% confidence)")
                else:
                    feedback_messages.append(f"❌ Incorrect pose detected ({confidence * 100:.1f}% confidence)")

                # Angle calculations (only if confidence is high)
                for angle_config in angle_configs:
                    try:
                        a = get_landmark_coordinates(results.pose_landmarks.landmark, angle_config["landmarks"][0])
                        b = get_landmark_coordinates(results.pose_landmarks.landmark, angle_config["landmarks"][1])
                        c = get_landmark_coordinates(results.pose_landmarks.landmark, angle_config["landmarks"][2])

                        angle = calculate_angle(a, b, c)
                        min_val, max_val = angle_config["desired_range"]

                        if not (min_val <= angle <= max_val):
                            feedback_messages.append(
                                f"Adjust {angle_config['name']}: Current {angle:.1f}° (Target: {min_val}-{max_val}°)"
                            )
                    except Exception as e:
                        feedback_messages.append(f"Could not calculate {angle_config['name']}")

                feedback_placeholder.info("\n".join(feedback_messages))

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        frame_placeholder.image(prepare_display_frame(frame, fps), channels="BGR")

    cap.release() 

# App routing
if "current_pose" not in st.session_state:
    main_page()
else:
    pose_page()
