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
            # Convert WebRTC frame to a PIL Image
            pil_image = frame.to_image()
            # Convert PIL Image to NumPy array
            img = np.array(pil_image)
            # Convert from RGB (PIL default) to BGR (OpenCV default)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Start with the original image as a default
            result_img = img

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.select_type == "Edge Detection":
                edges = cv2.Canny(gray, 50, 150)
                result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            elif self.select_type == "Face Detection":
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # --- THIS IS THE FIX ---
            # Convert the final BGR image (from OpenCV) back to RGB (for av)
            final_frame_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # Return an av.VideoFrame in 'rgb24' format
            return av.VideoFrame.from_ndarray(final_frame_rgb, format="rgb24")
            # ---------------------

        except Exception as e:
            print(f"Error in video processing: {e}")
            return frame


# 4. Main app UI
st.title("Video Capture Application")
st.write("This application captures video from your webcam and displays it in real-time.")

# 5. Run the WebRTC streamer
webrtc_streamer(
    key="video-capture",
    video_processor_factory=lambda: VideoProcessor(select_type=select_type),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.write("Click 'Start' to begin. You will need to grant camera permissions in your browser.")