import streamlit as st
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 1. Set page config
st.set_page_config(
    page_title="Video Capture App",
    page_icon=":movie_camera:",
    layout="centered",
    initial_sidebar_state="auto"
)

# 2. Set up the sidebar
st.sidebar.title("Settings")
select_type = st.sidebar.selectbox(
    "Select Detection Type",
    ("Edge Detection", "Face Detection")
)


# 3. Create the VideoProcessor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self, select_type):
        self.select_type = select_type
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        try:
            pil_image = frame.to_image()
            img = np.array(pil_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Create a copy for processing so we don't mess up the original visual
            result_img = img.copy()

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.select_type == "Edge Detection":
                # ONLY blur for edges. Use (3,3) or (5,5) for best balance
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            elif self.select_type == "Face Detection":
                # Do NOT blur here. Use the sharp 'gray' image.
                # Changed 4 to 6 for more stable detection (less flickering)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

                for (x, y, w, h) in faces:
                    # Draw a Green box (0, 255, 0) instead of white
                    # Thickness 2
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert back to RGB for display
            final_frame_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(final_frame_rgb, format="rgb24")

        except Exception as e:
            print(f"Error: {e}")
            return frame


# 4. Main app UI
st.title("Video Capture Application")
st.write("This application captures video from your webcam and displays it in real-time.")

# 5. Run the WebRTC streamer
webrtc_streamer(
    key="video-capture",
    video_processor_factory=lambda: VideoProcessor(select_type=select_type),
    # UPDATE: Request 1280x720 resolution for better quality
    media_stream_constraints={
        "video": {"width": 1280, "height": 720},
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.write("Click 'Start' to begin. You will need to grant camera permissions in your browser.")
