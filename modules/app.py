import streamlit as st
import quiz
import billboard_trend
import playlist
import artist_nation
import genre
import topic
import base64


# st.set_page_config를 최상단에서 한번만 호출
# st.set_page_config(page_title="Billboard", layout="centered")

# 로컬 이미지를 Base64로 인코딩하여 HTML에 삽입
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 로컬 이미지 경로 설정
background_image_path = "1jo/background.jpg"  # 여기에 로컬 이미지 경로를 입력하세요.

# 배경 이미지 적용
set_background_image(background_image_path)

# 사이드 메뉴에서 선택할 수 있는 페이지 목록
PAGES = {
    "Opening Quiz": quiz,
    "Where are they come from" : artist_nation,
    "What type of songs they sing" : genre,
    "Billboard_Trend": billboard_trend,
    "What are they singing about" : topic,
    "Make Your Playlist" : playlist
}

st.sidebar.image("1jo/Billboard_logo.svg", use_column_width=True)
# 사이드바에서 선택한 페이지로 이동
st.sidebar.title('Billboard Analysis')
selection = st.sidebar.radio("Menu", list(PAGES.keys()))

# 선택된 페이지 모듈을 호출하여 해당 페이지 실행
page = PAGES[selection]
page.app()
