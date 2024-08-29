import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import gensim
import nltk
import pyLDAvis.gensim
import os

# 불용어 확장
stop_words = set(stopwords.words('english'))

additional_stopwords = [
    'verse', 'chorus', 'i"ll', 'intro', 'outro', 'or', 'm', 'ma', 'ours', 'against', 'nor',
    'wasn', 'hasn', 'my', 'had', 'didn', 'isn', 'did', 'aren', 'those', 'than', 'man',
    "mustn't", "you've", 'to', 'she', 'having', "haven't", 'into', 't', 'll', 's','n',
    'himself', 'do', "that'll", 'so', 'of', 'on', 'very', 'for', 'out', 'were', '\'',
    'should', 'they', 'ain', "should've", 'you', "didn't", 'yours', 'was', 'our','ll',
    'can', 'myself', "shouldn't", 'have', 'up', 'mightn', "you'll", 'any', 't',
    'itself', 'hadn', 'him', 'doesn', 'weren', 'y', 'being', "don't", 'them', 
    'are', 'and', 'that', 'your', 'yourself', 'their', 'some', 'ourselves', 've', 
    'doing', 'been', 'shouldn', 'yourselves', "mightn't", 'most', 'because',
    'few', 'wouldn', "you'd", 'through', "you're", 'themselves', 'an', 'if',
    "wouldn't", 'its', 'other', "won't", "wasn't", "she's", 'we', 'shan',
    "weren't", 'don', "hadn't", 'this', 'off', 'while', 'a', 'haven', 'her', 
    'theirs', 'all', "hasn't", "doesn't", 'about', 'then', 'by', 'such', 'but', 
    'until', 'each', 'there', "aren't", 'with', 'not', "shan't", 'hers', 'it', 
    'too', 'i', 'at', 'is', 'as', 'me', 'herself', 's', 'the', 'where', 'am', 
    'has', 'over', "couldn't", 'when', 'does', 'mustn', 're', 'no', 'in', 'who', 
    'd', 'own', 'he', 'be', "isn't", 'his', 'these', 'same', 'whom', 'will', 
    'needn', 'couldn', 'from', "it's", 'o', 'yeah', 'ya', 'na', 'wan', 'uh', 'gon',
    'ima', 'mm', 'uhhuh', 'bout', 'em', 'nigga', 'niggas', 'got', 'ta', 'lil', 'ol', 'hey',
    'oooh', 'ooh', 'oh', 'youre', 'dont', 'im', 'youve', 'ive', 'theres', 'ill', 'yaka',
    'lalalala', 'la', 'da', 'di', 'yuh', 'shawty', 'oohooh', 'shoorah', 'mmmmmm',
    'ook', 'bidibambambambam', 'shh', 'bro', 'ho', 'aint', 'cant', 'know', 'bambam',
    'shitll', 'tonka','fn','uh', 'ah', 'oh', 'like', 'na','yeah','ai','chorus','let','verse','one','de','might','ca','bridge'
]

# spaCy 모델 로드

nlp = spacy.load('en_core_web_sm')
stop_words_spacy = set(nlp.Defaults.stop_words)
stop_words.update(additional_stopwords)
stop_words.update(stop_words_spacy)

# 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[df['Year'] != 1959]
    df['Decade'] = (df['Year'] // 10) * 10
    df['Decade'] = df['Decade'].astype(int)  # Decade 값을 정수형으로 변환
    df['Processed Lyrics'] = df['Lyrics'].apply(preprocess_lyrics)
    return df

def preprocess_lyrics(text):
    if isinstance(text, str):
        text = text.lower()
        words = word_tokenize(text)
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) >= 4]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words
    else:
        return []

# 캐싱된 데이터 로드
df = load_and_preprocess_data('1jo/all_songs_data.csv')
# LDA 모델 저장 경로 설정
model_dir = '1jo/'
os.makedirs(model_dir, exist_ok=True)


# LDA 모델을 학습하고 저장하는 함수
@st.cache_resource
def train_and_save_lda_model(decade_data, decade, num_topics=5, passes=15):
    dictionary = corpora.Dictionary(decade_data['Processed Lyrics'])
    corpus = [dictionary.doc2bow(text) for text in decade_data['Processed Lyrics']]
    
    model_path = os.path.join(model_dir, f'lda_model_{decade}.gensim')
    
    if os.path.exists(model_path):
        lda_model = gensim.models.ldamodel.LdaModel.load(model_path)
    else:
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        lda_model.save(model_path)
    
    return lda_model, corpus, dictionary

def app():
    # st.set_page_config(page_title="LDA Topic Visualization", layout="wide")
    st.title("🎵 Decade-wise Topic Visualization using LDA 🎵")
    
    # 사이드바 설정
    st.sidebar.title("Settings")
    selected_decade = st.sidebar.selectbox("Select a Decade:", sorted(df['Decade'].unique()))
    st.sidebar.markdown("This app allows you to explore topics in song lyrics across different decades.")
    
    decade_data = df[df['Decade'] == selected_decade]
    
    # 저장된 LDA 모델 불러오기 또는 학습하기
    lda_model, corpus, dictionary = train_and_save_lda_model(decade_data, selected_decade)
    
    # 토픽 시각화
    st.subheader(f"Top Words in Topic for {selected_decade}s")
    cols = st.columns(2)
    for i, topic in enumerate(lda_model.show_topics(num_topics=1, num_words=10, formatted=False)):
        words = [word for word, _ in topic[1]]
        topic_string = " ".join(words)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(topic_string)
        
        with cols[i % 2]:
            st.markdown(f"**Topic {i+1}**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    # LDA 시각화 준비
    st.subheader(f"LDA Topic Visualization for {selected_decade}s")
    lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda.html')
    st.components.v1.html(open('lda.html', 'r', encoding='utf-8').read(), height=800, width=1400)

# 앱 실행
if __name__ == "__main__":
    df = load_and_preprocess_data('1jo/all_songs_data.csv')
    model_dir = '1jo/'
    os.makedirs(model_dir, exist_ok=True)
    app()