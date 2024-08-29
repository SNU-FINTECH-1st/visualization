import streamlit as st
import pandas as pd
import nltk
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from math import pi
import numpy as np 
# 데이터 로드를 캐시 처리

# 필요할 수 있는 다른 NLTK 데이터도 설치
required_nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
for package in required_nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package)
def app():
    @st.cache_data
    def load_data():
        df = pd.read_csv('1jo/all_songs_data.csv')
        # punkt 데이터가 없으면 다운로드 시도
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        # 불용어 확장하기
        stop_words = set(stopwords.words('english'))  # 기본 영어 불용어 목록 로드
        stop_words_spacy = set(spacy.load('en_core_web_sm').Defaults.stop_words)
        stop_words_sklearn = set(ENGLISH_STOP_WORDS)
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
            'shitll', 'tonka','fn','ai'
        ]
    
        # 최종 불용어 통합
        stop_words = stop_words.union(stop_words_spacy).union(stop_words_sklearn).union(set(additional_stopwords))
    
        # 가사 전처리 함수 (리스트로 반환)
        def preprocess_lyrics(text):
            if isinstance(text, str):
                text = text.lower()
                words = word_tokenize(text)
                words = [word for word in words if word.isalpha() and word not in stop_words]
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words]
                return words  # 단어 리스트로 반환
            else:
                return []
    
        # 데이터프레임에 'processed_lyrics' 열 추가
        df['processed_lyrics_set'] = df['Lyrics'].apply(preprocess_lyrics)
        df['lyrics_for_emotion'] = df['Lyrics']
        return df
    
    # 감성 분석 모델 및 토크나이저 로드 (허깅페이스)
    @st.cache_resource
    def load_sentiment_model():
        sentiment_analyzer = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion', device=0)
        tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')
        return sentiment_analyzer, tokenizer
    
    # 데이터 로드
    df = load_data()
    sentiment_analyzer, tokenizer = load_sentiment_model()
    def plot_radar_chart(df):
        # 레이더 차트 설정
        categories = list(df.keys())
        values = list(df.values())
        
        N = len(categories)
        
        # 레이더 차트를 위한 각도 계산
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 레이더 차트를 닫기 위해 첫 번째 각도를 마지막에도 추가
        
        values += values[:1]  # 레이더 차트를 닫기 위해 첫 번째 값을 마지막에도 추가
    
        # 레이더 차트 그리기
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        # 차트의 배경, 그리드 및 틱 설정
        ax.set_facecolor('white')  # 배경색
        plt.xticks(angles[:-1], categories, color='darkblue', size=12)  # 카테고리 라벨 설정
        ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=0.5)  # 그리드 라인 스타일
        ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(angles[:-1], categories)
        
        ax.plot(angles, values)
        ax.fill(angles, values, 'b', alpha=0.1)
        # 레이더 차트 선 그리기
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
        ax.fill(angles, values, 'blue', alpha=0.3)  # 내부 채우기 색상과 투명도 설정
        # 최대값을 구하고 최대값보다 0.1 크게 설정
        max_value = max(values)
        plt.ylim(0, max_value + 0.1)
    
        # 동적으로 yticks 설정
        yticks = np.linspace(0, max_value + 0.1, num=6)  # 최대값에 따라 동적으로 yticks 생성
        plt.yticks(yticks, color="grey", size=10)
    
            # 중앙 원 스타일링
        ax.spines['polar'].set_color('darkblue')
        ax.spines['polar'].set_linewidth(1)
    
        st.pyplot(plt)
    
    # 애플리케이션 제목
    st.title('My Custom Playlist Builder 🎵')
    
    # 초기 검색어와 검색 옵션 설정
    search_lyrics = st.text_input("Enter lyrics or title to search for songs:", value="Christmas")
    search_option = st.radio("Search in:", ( 'Title','Lyrics'))
    
    # 플레이리스트 초기화
    if 'playlist' not in st.session_state:
        st.session_state.playlist = []
    
    # 선택된 노래를 저장할 상태 초기화
    if 'selected_songs' not in st.session_state:
        st.session_state['selected_songs'] = []
    
    # 빈 데이터프레임으로 초기화
    filtered_songs = pd.DataFrame()
    
    # 가사나 제목으로 노래 검색
    if search_lyrics:
        if search_option == 'Lyrics':
            filtered_songs = df[df['Lyrics'].str.contains(search_lyrics, case=False, na=False)]
        elif search_option == 'Title':
            filtered_songs = df[df['Song Title'].str.contains(search_lyrics, case=False, na=False)]
    
    # 검색 결과 표시
    st.subheader("Search Results")
    show_lyrics = st.checkbox("노래와 가사 보기")
    if not filtered_songs.empty and show_lyrics:
        button_out = "노래와 가사 리스트"
        
        for index, row in filtered_songs.iterrows():
            button_label = f"{row['Song Title']} by {row['Artist']} ({int(row['Year'])}) - Rank: {row['Rank']}"
            with st.expander(button_label):
                st.text_area(
                    f"Lyrics of {row['Song Title']} by {row['Artist']}",
                    row['Lyrics'],
                    height=200,
                    key=f"lyrics_{index}"
                )
    elif filtered_songs.empty :
        st.write("No results found. Please try a different search term.")
    else :
        st.write(" ")
    
    # 드래그 앤 드롭 또는 선택을 통해 플레이리스트에 추가
    st.subheader("Build Your Playlist")
    if not filtered_songs.empty:
        unique_songs = filtered_songs[['Song Title', 'Artist']].apply(lambda x: f"{x['Song Title']} by {x['Artist']}", axis=1).unique()
        selected_songs = st.multiselect('Select songs to add to your playlist:', unique_songs, default=st.session_state.selected_songs)
        st.session_state.selected_songs = selected_songs
    else:
        selected_songs = []
    
    # **Add to Playlist** 버튼을 사용하여 선택된 노래를 플레이리스트에 추가
    if st.button('Add to Playlist'):
        for song in st.session_state.selected_songs:
            if song not in st.session_state.playlist:
                st.session_state.playlist.append(song)
        st.session_state.playlist = list(set(st.session_state.playlist))  # 중복 제거
        st.success(f"Added {len(st.session_state.selected_songs)} songs to your playlist!")
        st.session_state.selected_songs.clear()
    
    # 플레이리스트 테이블 형식으로 표시 및 노래 제거 기능
    playlist_container = st.container()
    with playlist_container:
        if st.session_state.playlist:
            playlist_df = pd.DataFrame(st.session_state.playlist, columns=["Song"])
            playlist_df['Remove'] = playlist_df.index
            playlist_df.set_index('Song', inplace=True)
            st.table(playlist_df)
    
            # 플레이리스트에서 노래 제거
            for i, song in enumerate(st.session_state.playlist):
                if st.button(f"Remove {song}", key=f"remove_{i}"):
                    st.session_state.playlist.remove(song)
    
                    # 상태가 변경되었음을 표시하고, UI를 새로 고침하도록 함
                    with playlist_container:
                        st.experimental_set_query_params()  # 상태 변경 후 페이지를 새로 고침
                        st.session_state.playlist = st.session_state.playlist  # 상태 업데이트
                    break
    
    # 상태가 변경된 경우에만 재렌더링
    if 'reload' in st.session_state:
        st.session_state.pop('reload', None)  # 상태 제거하여 무한 루프 방지
    # 플레이리스트 분석
    if st.session_state.playlist:
        with st.container():  # 분석 결과와 워드 
            st.subheader("Playlist Analysis")
            
            # 플레이리스트의 가사 가져오기
            playlist_lyrics = df[df['Song Title'].isin([s.split(' by ')[0] for s in st.session_state.playlist])]['processed_lyrics_set'].sum()
            
            # 키워드 추출 (TF-IDF)
            st.markdown("### Playlist Keywords")
            tfidf_vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(playlist_lyrics)])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().flatten()
            tfidf_scores = pd.DataFrame({'Term': feature_names, 'Score': scores})
            tfidf_scores = tfidf_scores.sort_values(by='Score', ascending=False)
    
            # 워드클라우드 생성
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores.set_index('Term').to_dict()['Score'])
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    
            # 감성 분석
            st.markdown("### Sentiment Analysis")
    
            # 감성 분석에 사용할 텍스트를 최대 길이로 자르기 (모델 입력 크기 제한)
            max_length = 512
    
            # 각 곡별로 감성 점수를 계산
            sentiment_scores_list = []
    
            for lyric in playlist_lyrics:
                inputs = tokenizer(lyric, return_tensors='pt', truncation=True, max_length=max_length)
                truncated_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                
                # 감성 분석 (속도 개선을 위해 batch_size 사용)
                sentiments = sentiment_analyzer(truncated_text, batch_size=8)
                song_sentiment_scores = {label: 0 for label in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']}
                
                for sentiment in sentiments:
                    song_sentiment_scores[sentiment['label']] += sentiment['score']
    
                sentiment_scores_list.append(song_sentiment_scores)
    
            # 곡별 감성 점수를 평균하여 전체 감성 점수를 계산
            avg_sentiment_scores = {label: 0 for label in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']}
            
            for scores in sentiment_scores_list:
                for label, score in scores.items():
                    avg_sentiment_scores[label] += score
            avg_sentiment_scores['anger'] *= (2 / 3)
            
              # 평균 점수 계산
            avg_sentiment_scores = {label: score / len(sentiment_scores_list) for label, score in avg_sentiment_scores.items()}
    
            labels = list(avg_sentiment_scores.keys())
            values = list(avg_sentiment_scores.values())
    
            # 감정 분석 결과에서 가장 높은 감정을 찾기
            dominant_emotion = max(avg_sentiment_scores, key=avg_sentiment_scores.get)
            st.write(f"Your playlist's predominant emotion is **{dominant_emotion.capitalize()}**!")
            st.table(avg_sentiment_scores)
            plot_radar_chart(avg_sentiment_scores)
            #plot_radar_chart(avg_sentiment_scores)
            # 감정에 따른 이미지 표시
            emotion_images = {
                'sadness': '1jo/sadness.svg',
                'joy': '1jo/joy.svg',
                'love': '1jo/love.svg',
                'anger': '1jo/anger.svg',
                'fear': '1jo/fear.svg',
                'surprise': '1jo/surprise.svg'
            }
    
            if dominant_emotion in emotion_images:
                st.image(emotion_images[dominant_emotion], caption=f"Emotion: {dominant_emotion.capitalize()}", use_column_width=True)