import streamlit as st
import cv2
import time
st.set_page_config(
    page_title="Video Capture App",
    page_icon=":movie_camera:",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("Video Capture Application")
st.write("This application captures video from your webcam and displays it in real-time.")
#set sidebar having edge detection grayscale and face detection options
st.sidebar.title("Settings")
select_type = st.sidebar.selectbox(
    "Select Detection Type",
    ("Edge Detection", "Face Detection")
)

#create start and stop buttons
start_button = st.button("Start Video Capture")
stop_button = st.button("Stop Video Capture")
#placeholder for video frames
video_placeholder = st.empty()

if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if start_button:
    st.session_state.camera_running = True
if stop_button:
    st.session_state.camera_running = False
#video capture loop and edge detection of faces
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if select_type == "Edge Detection":
            #after detection convert to rgb edges and add button to click photo and save it in local directory
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            video_placeholder.image(edges, channels="RGB")
        elif select_type == "Face Detection":
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB")
            time.sleep(1/60)
    cap.release()
    video_placeholder.empty()
    st.write("Video capture stopped.")









