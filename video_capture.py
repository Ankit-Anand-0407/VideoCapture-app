import streamlit as st
import cv2
import av  # <-- 1. Import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase  # <-- 2. Import VideoProcessorBase

# Set page config
st.set_page_config(
    page_title="Video Capture App",
    page_icon=":movie_camera:",
    layout="centered",
    initial_sidebar_state="auto"
)

# Set up the sidebar
st.sidebar.title("Settings")
select_type = st.sidebar.selectbox(
    "Select Detection Type",
    ("Edge Detection", "Face Detection")
)


# 3. Create the *new* VideoProcessor class
class VideoProcessor(VideoProcessorBase):  # <-- 4. Use VideoProcessorBase
    def __init__(self, select_type):
        self.select_type = select_type
        # Load the cascade classifier ONCE
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 5. Use the `recv` method instead of `transform`
    def recv(self, frame):
        # Convert the frame from WebRTC to an array OpenCV can use
        img = frame.to_ndarray(format="bgr")

        # Common processing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.select_type == "Edge Detection":
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            # Convert grayscale edges back to 3-channel BGR
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif self.select_type == "Face Detection":
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            # Start with the original color image
            result_img = img
            for (x, y, w, h) in faces:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # 6. Return an `av.VideoFrame` instead of a numpy array
        return av.VideoFrame.from_ndarray(result_img, format="bgr")


# Main app UI
st.title("Video Capture Application")
st.write("This application captures video from your webcam and displays it in real-time.")

# Run the WebRTC streamer
webrtc_streamer(
    key="video-capture",
    # 7. Use `video_processor_factory`
    video_processor_factory=lambda: VideoProcessor(select_type=select_type),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.write("Click 'Start' to begin. You will need to grant camera permissions in your browser.")








