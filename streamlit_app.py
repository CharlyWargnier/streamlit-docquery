# https://colab.research.google.com/drive/1AzlWMXHwGNLtmi12YIOukddZLHjQttSO?usp=sharing#scrollTo=ZG495Ywkqs27
# https://huggingface.co/tasks/question-answering


import streamlit as st

import streamlit_nested_layout


# outer_cols = st.columns([1, 1])
#
# with outer_cols[0]:
#     st.markdown("## Column 1")
#     st.selectbox("selectbox", [1, 2, 3], key="sel1")
#
#     inner_cols = st.columns([1, 1])
#     with inner_cols[0]:
#         st.markdown("Nested Column 1")
#         st.selectbox("selectbox", [1, 2, 3], key="sel2")
#     with inner_cols[1]:
#         st.markdown("Nested Column 2")
#         st.selectbox("selectbox", [1, 2, 3], key="sel3")


# st.stop()


from transformers import pipeline
from PIL import Image


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

    # st.write(
    #     "This app relies on the new `document-question-answering` pipeline from the Huggingface `Transformers` library. It lets you get insights into your documents, invoices etc."
    # )
    #
    # st.write(
    #     "This model was created by the team at Impira. This is a fine-tuned version of the multi-modal LayoutLM model for the task of question answering on documents. It has been fine-tuned using both the SQuAD2.0 and DocVQA datasets. The model is available on the Huggingface model hub: https://huggingface.co/impira/layoutlm-document-qa"
    # )

    st.markdown("")

    st.markdown(
        """

    🎈 [Streamlit](https://streamlit.io/) app created by [Charly Wargnier](https://twitter.com/DataChaz) using [Impira's](https://www.impira.com/)'s DocQuery models.

    """
    )

    st.markdown(
        "DocQuery (created by [Impira](https://impira.com?utm_source=huggingface&utm_medium=referral&utm_campaign=docquery_space))"
        " uses LayoutLMv1 fine-tuned on DocVQA, a document visual question"
        " answering dataset, as well as SQuAD, which boosts its English-language comprehension."
        " To use it, simply upload an image or PDF, type a question, and click 'submit', or "
        " click one of the examples to load them."
        " DocQuery is MIT-licensed and available on [Github](https://github.com/impira/docquery)."
    )

    st.markdown("---")

_, left, right, _ = st.columns([1, 2, 2, 1], gap="medium")

with left:

    st.write("### 1. Select a file")

    st.caption("")

    file_name = st.file_uploader("", type=["png", "jpg", "jpeg"])
    c = st.container()

    outer_cols = st.columns([1, 1, 1, 1])

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

    with st.expander("Show document", expanded=True):

        if file_name is not None:
            image_sample = Image.open(file_name)
            st.image(image_sample, use_column_width=True)

    @st.experimental_singleton
    def get_pipeline():
        return pipeline("document-question-answering")

    pipe = get_pipeline()

    # pipe(image=image_sample, question="what is the purchase amount?")

with right:

    st.write("### 2. Ask a question")

    st.caption("E.g.: What is the purchase amount? What is the date of the invoice?")

    with st.form(key="my_form"):

        CHECKPOINTS = {
            "LayoutLMv1": "impira/layoutlm-document-qa",
            "LMv1/Invoices": "impira/layoutlm-invoices",
            "Donut 🍩": "naver-clova-ix/donut-base-finetuned-docvqa",
        }

        model = st.radio("Select a model", list(CHECKPOINTS.keys()), horizontal=True)

        question = st.text_input(
            label="Question", placeholder="What is the purchase amount?"
        )

        submit_button = st.form_submit_button(label="Submit")

        # question = st.text_input("Question", value="what is the purchase amount?")

    if file_name is not None:

        if submit_button:

            answer = pipe(image=image_sample, question=question)
            st.write(answer)

    else:
        c.info("☝️ Please upload a document to get started.")

        st.stop()
