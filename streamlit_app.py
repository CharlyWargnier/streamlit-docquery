# https://colab.research.google.com/drive/1AzlWMXHwGNLtmi12YIOukddZLHjQttSO?usp=sharing#scrollTo=ZG495Ywkqs27

import streamlit as st
from transformers import pipeline


st.image(
    # "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/hugging-face_1f917.png", width=100)
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/sparkles_2728.png",
    width=100,
)

st.title("DocQuery app")

st.write(
    "This app uses the `pipeline` from the `transformers` library to query a document."
)

# You can use a http link, a local path or a PIL.Image object
img_path = "contract.jpeg"

with st.expander("Show image"):
    st.image(img_path, width=1000)

# pipe = pipeline("document-question-answering")

@st.experimental_singleton
def get_pipeline():
    return pipeline("document-question-answering")

pipe = get_pipeline()

pipe(image=img_path, question="what is the purchase amount?")

with st.form(key="my_form"):
    question = st.text_input(label="Enter your question")
    submit_button = st.form_submit_button(label="Submit")


# question = st.text_input("Question", value="what is the purchase amount?")

st.write(pipe(image=img_path, question=question))
