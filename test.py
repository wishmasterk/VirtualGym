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
model_path = 'EfficientNetB0.keras'
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


def pose_page():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .pose-header {
        font-size: 2.5rem !important;
        color: #2e86ab;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f18f01;
    }
    .pose-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pose-section h3 {
        color: #2e86ab !important;
    }
    .benefit-badge {
        background-color: #a5d8ff !important;
        color: #1864ab !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main layout columns
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Pose Header with Icon
        st.markdown(f"""
        <div class='pose-header'>
            <i class='fas fa-tree' style='margin-right:10px;'></i>
            {st.session_state.current_pose.replace("_", " ")}
        </div>
        """, unsafe_allow_html=True)
        
        # Sample Image Card
        with st.container():
            pose_info = POSE_INFO[st.session_state.current_pose]
            try:
                st.image(pose_info.get("sample_image", ""), 
                        caption="PROPER FORM EXAMPLE",
                        use_container_width=True)
            except:
                st.warning("Sample image not available")
            
            if st.button("↩ Back to Main Menu", 
                        use_container_width=True,
                        type="primary"):
                del st.session_state.current_pose
                st.rerun()

    with col2:
        # Description Card
        with st.container():
            st.markdown("""
            <div class='pose-section'>
                <h3>🧘 About This Pose</h3>
                <p>{}</p>
            </div>
            """.format(pose_info["description"]["About"]), 
            unsafe_allow_html=True)
            
            # Benefits Card
            st.markdown("""
            <div class='pose-section'>
                <h3>✨ Benefits</h3>
                <div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>
            """, unsafe_allow_html=True)
            
            for benefit in pose_info["description"]["Benefits"]:
                st.markdown(f"""
                <span class='benefit-badge' style='padding: 0.3rem 1rem; border-radius: 100px;'>
                    {benefit}
                </span>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Steps & Guide Cards in Tabs
            tab1, tab2, tab3 = st.tabs(["📝 Steps", "⚠️ Common Mistakes", "💡 App Guide"])
            
            with tab1:
                # Create HTML list items
                steps_html = "".join([f"<li>{step}</li>" for step in pose_info["description"]["Steps"]])
                
                st.markdown(f"""
                <div class='pose-section'>
                    <h3>Step-by-Step !!</h3>
                    <ul>
                        {steps_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                # Create HTML list items
                steps_html = "".join([f"<li>{step}</li>" for step in pose_info["description"]["Pose Errors"]])
                
                st.markdown(f"""
                <div class='pose-section'>
                    <h3>Avoid These Mistakes !!</h3>
                    <ul>
                        {steps_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            with tab3:
                # Create HTML list items
                steps_html = "".join([f"<li>{step}</li>" for step in pose_info["description"]["App Guide"]])
                
                st.markdown(f"""
                <div class='pose-section'>
                    <h3>What To Do !!</h3>
                    <ul>
                        {steps_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_placeholder = st.empty()
    feedback_placeholder = st.empty()
    target_class = POSE_INFO[st.session_state.current_pose]["class_index"]
    angle_configs = POSE_INFO[st.session_state.current_pose]["angles"]

    # Add frame skip counter
    frame_counter = 0
    FRAME_SKIP = 15  # Process every nth frame for predictions
    last_feedback = []

    # FPS calculation variables
    fps_start_time = time.time()
    frames_this_second = 0
    current_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Count frames and update FPS every second
        current_time = time.time()
        frames_this_second += 1
        
        # If one second has passed, update the FPS count
        if current_time - fps_start_time >= 1.0:
            current_fps = frames_this_second
            frames_this_second = 0
            fps_start_time = current_time
        
        # Person detection check - only on frames we process
        if frame_counter % FRAME_SKIP == 0:
            person_valid, person_message = contains_person(frame)
        
            if not person_valid:
                fps = 1 / (time.time() - fps_start_time)
                frame_placeholder.image(prepare_display_frame(frame, fps), channels="BGR")
                if person_message:
                    feedback_placeholder.warning(person_message)
                frame_counter += 1
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

            # Only process predictions and angles every FRAME_SKIP frames
            if frame_counter % FRAME_SKIP == 0:
                # Model prediction
                input_frame = cv2.resize(frame, (224, 224)) / 255.0
                prediction = model_pose.predict(np.expand_dims(input_frame, axis=0), verbose=0)
                confidence = np.max(prediction, axis=1)[0]
                predicted_class = np.argmax(prediction, axis=1)[0]
                last_prediction = (predicted_class, confidence)

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

                last_feedback = feedback_messages
                feedback_placeholder.info("\n".join(last_feedback))
            else:
                # Use the last feedback for non-processed frames
                if last_feedback:
                    feedback_placeholder.info("\n".join(last_feedback))

        # Calculate and display FPS
        display_frame = prepare_display_frame(frame, current_fps)
        frame_placeholder.image(display_frame, channels="BGR")

        frame_counter += 1

    cap.release() 

def prepare_display_frame(frame, fps):
    display_frame = cv2.resize(frame, (640, 480))
    # Add "Average" to the FPS display and format to 1 decimal place
    cv2.putText(display_frame, f"Average FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return display_frame

# App routing
if "current_pose" not in st.session_state:
    main_page()
else:
    pose_page()
