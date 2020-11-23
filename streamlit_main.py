import streamlit as st
import pandas as pd
import spacy


def get_result(score):
    if score["Negative"] > score["Positive"]:
        return "Negative"
    else:
        return "Positive"


st.title('Sentiment/Content Analysis')
nlp = spacy.load("/Users/tanushnadimpalli/Documents/python_stuff_new")

text = st.text_input(label="Type a sentence to run analysis on")

if text:
    doc = nlp(text)

    negative = doc.cats["Negative"]
    positive = doc.cats["Positive"]

    data = {'Value': [negative, positive]}
    df = pd.DataFrame(data)
    df.index = ["Negative", "Positive"]

    st.subheader("Result :")

    result = get_result(doc.cats)
    st.write(result)

    st.subheader("Label Scores :")

    st.write(df)
    st.bar_chart(data=df, use_container_width=True)
