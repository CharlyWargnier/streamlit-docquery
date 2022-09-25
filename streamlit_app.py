# https://colab.research.google.com/drive/1AzlWMXHwGNLtmi12YIOukddZLHjQttSO?usp=sharing#scrollTo=pOfOs4lzqpJp
# https://huggingface.co/docs/transformers/main_classes/pipelines

import streamlit as st
import streamlit_nested_layout
from transformers import pipeline
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="DocQuery: Document Query Engine",
    page_icon="🍩",
    layout="wide",
)

_, centre, _ = st.columns([1, 4, 1])

with centre:

    st.image(
        "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/doughnut_1f369.png",
    )

    st.title("DocQuery: Document Query Engine")

    st.markdown("")

    st.markdown(
        """

    🎈 [Streamlit](https://streamlit.io/) app created by [Charly Wargnier](https://twitter.com/DataChaz) using [Impira](https://www.impira.com/)'s DocQuery models.

    """
    )

    st.markdown(
        "DocQuery uses [LayoutLMv1](https://huggingface.co/docs/transformers/model_doc/layoutlm) fine-tuned on [DocVQA](https://www.docvqa.org/), a document visual question"
        " answering dataset, as well as The Stanford Question Answering Dataset ([SQuAD](https://paperswithcode.com/dataset/squad)), which boosts its English-language comprehension."
    )

    st.markdown(
        "To use it, simply upload a document in `png` or `jpeg` format, type a question, and click 'Submit'!✨"
    )

    st.markdown("---")

_, left, right, _ = st.columns([1, 2, 2, 1], gap="medium")

with left:

    st.write("### 1. Select a file")

    st.caption("")

    file_name = st.file_uploader("", type=["png", "jpg", "jpeg"])
    c = st.container()

    st.caption("")
    st.caption("Download sample files")

    outer_cols = st.columns([0.9, 1, 1, 1])

    with outer_cols[0]:

        with open("contract.jpeg", "rb") as file:
            btn = st.download_button(
                label="Contract",
                data=file,
                file_name="contract.jpeg",
                mime="image/png",
                key="download_1",
            )

    with outer_cols[1]:

        with open("statement.png", "rb") as file:
            btn = st.download_button(
                label="Statement",
                data=file,
                file_name="statement.png",
                mime="text/csv",
                key="download_2",
            )

    with outer_cols[2]:

        with open("invoice.png", "rb") as file:
            btn = st.download_button(
                label="Invoice",
                data=file,
                file_name="invoice.png",
                mime="text/csv",
                key="download_3",
            )

    st.write("")

    #    with st.expander("Show document", expanded=True):

    if file_name is not None:
        image_sample = Image.open(file_name)
        st.image(image_sample, use_column_width=True)


with right:

    st.write("### 2. Ask a question")

    st.caption("Eg: What is the purchase amount? What is the invoice date?")

    with st.form(key="my_form"):

        CHECKPOINTS = {
            "LMv1": "impira/layoutlm-document-qa",
            "LMv1/Invoices": "impira/layoutlm-invoices",
            "Donut 🍩": "naver-clova-ix/donut-base-finetuned-docvqa",
        }

        model = st.radio("Select a model", list(CHECKPOINTS.keys()), horizontal=True)

        question = st.text_input(
            label="Question", placeholder="What is the purchase amount?"
        )

        submit_button = st.form_submit_button(label="Submit")

    if file_name is not None:

        if submit_button:

            @st.cache(allow_output_mutation=True)
            def get_pipeline():
                return pipeline("document-question-answering", model=CHECKPOINTS[model])

            pipe = get_pipeline()

            answer = pipe(image=image_sample, question=question)
            answer
            score = answer["score"]
            score = "{:.2%}".format(score)

            answers = answer["answer"]
            start = answer["start"]
            end = answer["end"]

            st.caption("")

            st.write("### 3. Check the answer")

            # bootstrap 4 collapse example
            components.html(
                f"""
            <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">

            <div class="ui vertical steps">
            <div class="active step">
                <i class="check square outline icon"></i>
                <div class="content">
                <div class="title">Answer</div>
                <div class="description">{answers}</div>
                </div>
            </div>
            <div class="active step">
                <i class="percent icon"></i>
                <div class="content">
                <div class="title">Score</div>
                <div class="description">{score}</div>
                </div>
            </div>
            <div class="active step">
                <i class="play icon"></i>
                <div class="content">
                <div class="title">Start</div>
                <div class="description">{start}</div>
                </div>
            </div>
            <div class="active step">
                <i class="stop circle icon"></i>
                <div class="content">
                <div class="title">End</div>
                <div class="description">{end}</div>
                </div>
            </div>
            </div>

                """,
                height=500,
            )

            st.write("### ImageDraw Drafts!")

            import sys
            from PIL import Image, ImageDraw

            highlight = image_sample.copy()
            draw = ImageDraw.Draw(highlight)
            draw.rectangle([start, end, end, start], outline="red", width=5)
            st.image(highlight)

    else:
        c.info("☝️ Please upload a document to get started.")

        st.stop()
