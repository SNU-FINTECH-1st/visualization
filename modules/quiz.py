import streamlit as st

def app():
    # Page title and subtitle styling
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 3em;
                color: #FF5733;
                font-weight: bold;
                text-align: center;
                margin-bottom: 0;
            }
            .sub-title {
                font-size: 1.5em;
                color: #C70039;
                text-align: center;
                margin-top: 0;
                margin-bottom: 40px;
            }
            .answer-title {
                font-size: 1.5em;
                color: #581845;
                font-weight: bold;
            }
            .answer-text {
                font-size: 1.2em;
                color: #2C3E50;
            }
            .footer-text {
                font-size: 1.5em;
                color: #2C3E50;
                text-align: center;
                margin-top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 class='main-title'>🎶  K-Pop Quiz: Which Korean Artists Made it to the Billboard Top 100?</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Select a difficulty level to reveal the artist and their song!</div>", unsafe_allow_html=True)
    
    # Answer list with difficulty, YouTube links, and start time
    answers = {
        "BTS": {"difficulty": 1, "youtube_url": "https://www.youtube.com/embed/WMweEpGlu_U", "start_time": 35},
        "Psy": {"difficulty": 2, "youtube_url": "https://www.youtube.com/embed/9bZkp7q19f0", "start_time": 68},
        "Fifty Fifty": {"difficulty": 3, "youtube_url": "https://www.youtube.com/embed/Qc7_zRjH808", "start_time": 40},
        "Pinkfong": {"difficulty": 4, "youtube_url": "https://www.youtube.com/embed/761ae_KDg_Q", "start_time": 27}
    }
    
    # Add a default "Choose a difficulty" option
    difficulty_options = ["Choose a difficulty level"] + [f"Difficulty {info['difficulty']}" for artist, info in sorted(answers.items(), key=lambda x: x[1]["difficulty"])]
    selected_difficulty = st.selectbox("Choose a difficulty level:", options=difficulty_options)

    # Display answer based on selected difficulty
    if selected_difficulty != "Choose a difficulty level":
        artist = next(artist for artist, info in answers.items() if f"Difficulty {info['difficulty']}" == selected_difficulty)
        info = answers[artist]
        start_time = info["start_time"]
        youtube_url_with_start = f"{info['youtube_url']}?start={start_time}&autoplay=1"
        st.markdown(f"<div class='answer-title'>🎉 Answer: **{artist}**</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-text'>{artist} is a Korean artist who made it to the Billboard Top 100.</div>", unsafe_allow_html=True)
        # Embed YouTube video with start time and autoplay
        st.markdown(f"""
            <iframe width="700" height="400" src="{youtube_url_with_start}" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
        """, unsafe_allow_html=True)
    
    # Footer information
    st.markdown("""
    <hr style='border: 1px solid #f0f0f0;'>
    <div class='footer-text'>
        <h3>We have learned about the Korean artists who reached the Billboard Top 100 through this quiz.</h3>
        <h4>Let's begin the presentation now!</h4>
    </div>
    """, unsafe_allow_html=True)
    
if __name__ == '__main__':
    app()
