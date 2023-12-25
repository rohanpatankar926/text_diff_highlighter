import streamlit as st
from module import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics import edit_distance
from nltk.translate.bleu_score import sentence_bleu

def compare_texts(actual_text, predicted_text):
    matcher = SequenceMatcher(None, actual_text.split(), predicted_text.split())
    differences = matcher.get_opcodes()

    actual_value = []
    predicted_value = []
    for tag, i1, i2, j1, j2 in differences:
        if tag == 'replace':
            actual_value.extend(actual_text.split()[i1:i2])
            predicted_value.extend(predicted_text.split()[j1:j2])

    return actual_value, predicted_value

def highlight_differences(actual_text, predicted_text, actual_value, predicted_value):
    highlighted_text = []

    for a, p in zip(actual_text.split(), predicted_text.split()):
        if a.lower() != p.lower():
            highlighted_text.append(f'<span style="color:red">{p}</span>')
        else:
            highlighted_text.append(p)

    highlighted_predicted = ' '.join(highlighted_text)
    return actual_value, predicted_value, highlighted_predicted

def calculate_cosine_similarity(actual_text, predicted_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([actual_text, predicted_text])
    cosine_similarity = (vectors * vectors.T).A[0, 1]
    return cosine_similarity

def calculate_word_error_rate(actual_text, predicted_text):
    wer = edit_distance(actual_text.split(), predicted_text.split())
    wer_rate = wer / len(actual_text.split())
    return wer_rate

def calculate_bleu_score(actual_text, predicted_text):
    reference = [actual_text.split()]
    candidate = predicted_text.split()
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

st.title('Text Comparison App')

actual_text = st.text_area('Enter Actual Text:')
predicted_text = st.text_area('Enter Predicted Text:')

if st.button('Compare Texts'):
    actual, predicted = compare_texts(actual_text, predicted_text)
    actual, predicted, highlighted_output = highlight_differences(actual_text, predicted_text, actual, predicted)
    
    st.markdown(f'<span style="color:blue">**Actual Value:**</span> {actual_text}', unsafe_allow_html=True)
    st.markdown(f'<span style="color:blue">**Predicted Value:**</span> {highlighted_output}', unsafe_allow_html=True)
    st.write(f'<span style="color:blue">**Actual List:**</span> {actual}', unsafe_allow_html=True)
    st.write(f'<span style="color:blue">**Predicted List:**</span> {predicted}', unsafe_allow_html=True)
    
    cosine_similarity = calculate_cosine_similarity(actual_text, predicted_text)
    st.markdown(f'**Cosine Similarity:** <span style="color:green">{cosine_similarity}</span>', unsafe_allow_html=True)
    
    wer_rate = calculate_word_error_rate(actual_text, predicted_text)
    st.markdown(f'**Word Error Rate:** <span style="color:orange">{wer_rate}</span>', unsafe_allow_html=True)
    
    bleu_score = calculate_bleu_score(actual_text, predicted_text)
    st.markdown(f'**BLEU Score:** <span style="color:yellow">{bleu_score}</span>', unsafe_allow_html=True)