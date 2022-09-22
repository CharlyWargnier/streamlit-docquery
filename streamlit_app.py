# https://colab.research.google.com/drive/1AzlWMXHwGNLtmi12YIOukddZLHjQttSO?usp=sharing#scrollTo=ZG495Ywkqs27

import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(
    page_title="QueryDoc",
    page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/magnifying-glass-tilted-right_1f50e.png",
    layout="centered",
)

st.image(
    # "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/hugging-face_1f917.png", width=100)
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/magnifying-glass-tilted-right_1f50e.png",
)

st.title("QueryDoc")
st.write("")

st.write(
    "This app relies on the new `document-question-answering` pipeline from the Huggingface `Transformers` library. It lets you get insights into your documents, invoices etc."
)

file_name = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg"])

with st.expander("Show document"):

    if file_name is not None:
        image_sample = Image.open(file_name)
        st.image(image_sample, use_column_width=True)


@st.experimental_singleton
def get_pipeline():
    return pipeline("document-question-answering")


pipe = get_pipeline()

# pipe(image=image_sample, question="what is the purchase amount?")

with st.form(key="my_form"):
    question = st.text_input(label="Enter your question")
    submit_button = st.form_submit_button(label="Submit")

# question = st.text_input("Question", value="what is the purchase amount?")

if file_name is not None:
    if submit_button:
        st.write(pipe(image=image_sample, question=question))
else:
    st.info("☝️ Please upload a document format (pdf, png, jpg, jpeg)")
    st.stop()
