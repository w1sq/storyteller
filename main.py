import cv2
import numpy
import streamlit
import tempfile
from gradio_client import Client

streamlit.write("Welcome to our website!")
streamlit.write("Drop a video below to get a story based on random frame of it")
uploaded_file = streamlit.file_uploader("Choose a video...", type=["mp4", "mov"])
client = Client("https://tonyassi-image-story-teller.hf.space/--replicas/liw84/")


def get_random_frame(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        f.write(video_bytes)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = numpy.random.randint(0, frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()
    frame = get_random_frame(video_bytes)
    streamlit.image(frame, caption="Frame to generate story from")

    with streamlit.spinner("Loading..."):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
            cv2.imwrite(temp_file.name, frame)
            temp_file.flush()

            result = client.predict(
                temp_file.name,
                api_name="/predict",
            )

    streamlit.success("Here is your story:")
    streamlit.write(result)
