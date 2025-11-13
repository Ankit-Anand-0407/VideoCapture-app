import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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


# 3. Create the VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self, select_type):
        self.select_type = select_type
        # Load the cascade classifier ONCE when the class is initialized
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        # Convert the frame from WebRTC to an array OpenCV can use
        img = frame.to_ndarray(format="bgr")

        # Common processing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.select_type == "Edge Detection":
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            # Convert grayscale edges back to 3-channel BGR to display
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif self.select_type == "Face Detection":
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles on the *original BGR* image
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            result_img = img

        # Return the processed BGR image
        return result_img


# 4. Main app UI
st.title("Video Capture Application")
st.write("This application captures video from your webcam and displays it in real-time.")

# 5. Run the WebRTC streamer
webrtc_streamer(
    key="video-capture",
    # Pass the selected detection type to the transformer
    video_transformer_factory=lambda: VideoTransformer(select_type=select_type),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.write("Click 'Start' to begin. You will need to grant camera permissions in your browser.")








