import logging
import threading
import urllib.request
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import mediapipe as mp

import streamlit as st

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("WebRTC demo")

    app_lip_and_face_detection_page = "Detect lip and face"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            app_lip_and_face_detection_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == app_lip_and_face_detection_page:
        app_lip_and_face_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")



def app_lip_and_face_detection():
    """Detect facial expressions and
    lip movements
    """
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()


    class FaceDetectionVideoProcessor(VideoProcessorBase):

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Facial landmarks
            result = face_mesh.process(rgb_image)
            height, width, _ = img.shape
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv2.circle(img, (x, y), 2, (100, 100, 0), -1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FaceDetectionVideoProcessor,
        async_processing=True,
    )

    st.markdown("Detect face and lips")


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
