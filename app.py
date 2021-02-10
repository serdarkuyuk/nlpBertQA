import os
import streamlit as st
import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

name = "mrm8488/bert-small-finetuned-squadv2"

tokenizer = AutoTokenizer.from_pretrained(name,)

model = AutoModelForQuestionAnswering.from_pretrained(name)


def welcome():
    return "Welcome All"


def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer` string and tries to identify
    the words within the `answer` that can answer the question. Prints them out.
    '''

    # tokenize the input text and get the corresponding indices
    token_indices = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(token_indices)))

    # Search the input_indices for the first instance of the `[SEP]` token.
    sep_index = token_indices.index(tokenizer.sep_token_id)

    seg_one = sep_index + 1

    # The remainders lie in the second segment.
    seg_two = len(token_indices) - seg_one

    # Construct the list of 0s and 1s.
    segment_ids = [0]*seg_one + [1]*seg_two

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(token_indices)

    # get the answer for the question
    start_scores, end_scores = model(torch.tensor([token_indices]),  # The tokens representing our input combining question and answer.
                                     # The segment IDs to differentiate question from answer
                                     token_type_ids=torch.tensor([segment_ids]),
                                     return_dict=False)
    # print(start_scores)
    # Find the tokens with the highest `start` and `end` scores.
    answer_begin = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    indices_tokens = tokenizer.convert_ids_to_tokens(token_indices)

    answer = indices_tokens[answer_begin:answer_end+1]
    # remove special tokens
    # use this when using model "twmkn9/albert-base-v2-squad2"
    answer = [word.replace("▁", "") if word.startswith("▁") else word for word in answer]
    answer = " ".join(answer).replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "")

    return answer


def example():

    answer_text = "Just A Rather Very Intelligent System (J.A.R.V.I.S.) was originally Tony Stark’s natural-language user interface computer system, named after Edwin Jarvis, the butler who worked for Howard Stark. Over time, he was upgraded into an artificially intelligent system, tasked with running business for Stark Industries as well as security for Tony Stark’s Mansion and Stark Tower. After creating the Mark II armor, Stark uploaded J.A.R.V.I.S. into all of the Iron Man Armors, as well as allowing him to interact with the other Avengers, giving them valuable information during combat."
    question = "who created mark II"

    return question, answer_text


def main():
    st.title("QA Prediction From a Text")

    st.text('This app is QA analysis project usint tranformer architecture.  For detail information follow the below link.')
    link = '[GitHub](https://github.com/serdarkuyuk/nlpBertQA)'
    st.markdown(link, unsafe_allow_html=True)
    #st.text_input("Text-Email", "Type your email here.")
    st.header('Example how to use this website')
    question, answer_text = example()

    st.text("Let's assume you have this text ")
    answer_text = st.text_area("Example text", answer_text)

    st.text('You have asked this question')
    question = st.text_area("Example question", question)

    if st.button("Example Answer"):
        answer = answer_question(question, answer_text)
        st.text('The model has an answer')
        st.success(answer.capitalize())
        st.text('If you like it please send an email to serdarkuyuk@gmail.com')

    st.header('Now it is your turn')
    answer_text = st.text_area("Your text here", "Type your sentence here.")
    question = st.text_area("Your question here", "Type your sentence here.")

    if st.button("Answer"):
        answer = answer_question(question, answer_text)
        st.success(answer.capitalize())
        st.text('If you like it please send an email to serdarkuyuk@gmail.com')

        return answer


if __name__ == '__main__':

    main()
    # Wrap text to 80 characters.
    #wrapper = textwrap.TextWrapper(width=80)

    # bert_abstract = "ust A Rather Very Intelligent System (J.A.R.V.I.S.) was originally Tony Stark’s natural-language user interface computer system, named after Edwin Jarvis, the butler who worked for Howard Stark. Over time, he was upgraded into an artificially intelligent system, tasked with running business for Stark Industries as well as security for Tony Stark’s Mansion and Stark Tower. After creating the Mark II armor, Stark uploaded J.A.R.V.I.S. into all of the Iron Man Armors, as well as allowing him to interact with the other Avengers, giving them valuable information during combat."

    # print(wrapper.fill(bert_abstract))

    # question, answer_text = example()
    # answer = answer_question(question, answer_text)
    # print(answer)
