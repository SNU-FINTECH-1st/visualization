import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
import yt_dlp  # yt-dlp for YouTube video extraction
import streamlit.components.v1 as components

def app():
    st.title("Billboard Trend Page")
    st.write("Welcome to the Billboard Trend Page.")
    # Additional Billboard trend analysis features can be added here.

    # Load DataFrame
    df = pd.read_csv('1jo/billboard_24years_lyrics_spotify.csv') 
    week_100 = pd.read_csv('1jo/hot-100-current.csv')

    # Filter data where 'current_week' equals 1 (only #1 positions)
    top1_data = week_100[week_100['current_week'] == 1]
    
    # Streamlit App Start
    st.title("Billboard Chart #1 Artist Analysis")
    
    # Set default selected artists
    default_artists = ['Billy Joel', 'Adele', 'Bee Gees', 'Bruno Mars']
    
    # Artist selection (sorted alphabetically, multiple selection enabled)
    selected_artists = st.multiselect(
        "Select artists:", 
        options=sorted(top1_data['performer'].unique()),
        default=default_artists
    )
    
    if selected_artists:
        # Calculate #1 count for selected artists
        selected_data = top1_data[top1_data['performer'].isin(selected_artists)]
        
        # Count unique performer-title combinations to calculate #1 count
        artist_song_count = selected_data.groupby(['performer', 'title']).size().reset_index(name='count')
        artist_count = artist_song_count.groupby('performer')['count'].sum().reset_index()
    
        # Treemap Visualization
        fig = px.treemap(
            artist_count,
            path=['performer'],
            values='count',
            color='count',
            color_continuous_scale='Blues',
            title='Number of Billboard #1s by Artist'
        )
    
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig)
    else:
        st.write("Results will be displayed here when artists are selected.")

    
    # Apply Seaborn Style
    sns.set(style="whitegrid")

    
    st.title("📈 Billboard Trend Change Visualization")
    st.write("Interactive analysis of Billboard Hot 100 data.")
    
    # Set default to 2023
    years = pd.to_datetime(week_100['chart_week']).dt.year.unique()
    selected_year = st.selectbox("Select a Year", sorted(years), index=list(years).index(2023) if 2023 in years else 0)
    
    # Process data for the selected year
    year_data = week_100[pd.to_datetime(week_100['chart_week']).dt.year == selected_year]
    
    # Contribution Score Calculation Function
    def calculate_contribution_score(df, year):
        df_year = df[pd.to_datetime(df['chart_week']).dt.year == year]
        df_year['score'] = 101 - df_year['current_week']
        contribution = df_year.groupby('performer')['score'].sum().reset_index()
        return contribution.sort_values(by='score', ascending=False)
    
    # Calculate contribution scores for each artist
    contribution_df = calculate_contribution_score(week_100, selected_year)
    
    # 🎨 Word Cloud Section
    with st.expander("📊 Word Cloud & Top Trends"):
        st.markdown(f"### 🎤 Artists' Contribution Word Cloud in {selected_year}")
        word_freq = dict(zip(contribution_df['performer'], contribution_df['score']))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='tab20',
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        st.image(wordcloud.to_array(), use_column_width=True)
    
    # 📊 Trend Graph Section
    with st.expander("🎵 Top Songs Rank Trends"):
        st.markdown(f"### Top 10 Songs Rank Trend in {selected_year}")
        top_songs_data = year_data.groupby(['title', 'performer']).min().sort_values(by='peak_pos', ascending=True).head(10)
        top_songs_data = year_data[year_data.set_index(['title', 'performer']).index.isin(top_songs_data.index)]
    
        # Interactive Graph with Plotly
        fig = px.line(
            top_songs_data, 
            x='chart_week', 
            y='current_week', 
            color='title', 
            line_group='performer',
            hover_name='title',
            hover_data={
                'performer': True,
                'current_week': True,
                'title': False  # title is already in hover_name
            },
            markers=True,
            labels={
                'chart_week': 'Week',
                'current_week': 'Rank',
                'title': 'Song Title',
                'performer': 'Performer'
            },
            title=f"Top 10 Songs Rank Trend in {selected_year}"
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.4,
                xanchor="center",
                x=0.5
            ),
            height=600  # Increase graph height
        )
        fig.update_yaxes(autorange="reversed")  # Reverse the y-axis since lower rank is better
        st.plotly_chart(fig, use_container_width=True)
    
    # 🎶 Artist and Song Selection Section
    selected_singer = st.selectbox("Select a Singer", contribution_df['performer'].tolist())
    
    if selected_singer:
        st.markdown(f"### {selected_singer}'s Songs on the Chart")
    
        # Retrieve all charting songs for the artist across all years
        singer_data = week_100[week_100['performer'].str.contains(selected_singer)]
        singer_data_agg = singer_data.groupby('title').agg({
            'peak_pos': 'min',
            'wks_on_chart': 'max',
            'chart_week': 'min'  # First chart entry date
        }).reset_index().sort_values(by='title')
    
        # Rename columns
        singer_data_agg.rename(columns={'chart_week': 'first_chart_in'}, inplace=True)
        
        st.dataframe(singer_data_agg, width=800)
    
        # Add song selection box
        selected_song = st.selectbox("Select a Song", singer_data_agg['title'].tolist())
        
        if selected_song:
            st.markdown(f"### Weekly Rank Trend for '{selected_song}' by {selected_singer}")
            
            song_week_data = singer_data[singer_data['title'] == selected_song].sort_values(by='chart_week')  # Sort by date
            
            if not song_week_data.empty:
                fig = px.line(
                    song_week_data, 
                    x='chart_week', 
                    y='current_week', 
                    markers=True,
                    title=f"Weekly Rank Trend for '{selected_song}' by {selected_singer}",
                    labels={
                        'chart_week': 'Week',
                        'current_week': 'Rank'
                    },
                    hover_data={
                        'chart_week': True,
                        'current_week': True
                    }
                )
                fig.update_layout(
                    height=600,  # Increase graph height
                    margin=dict(l=0, r=0, t=50, b=50)
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
    
                # YouTube search and video embedding
                search_query = f"{selected_singer} {selected_song}"
                ydl_opts = {
                    'format': 'best',
                    'noplaylist': True,
                    'quiet': True,
                    'default_search': 'ytsearch1',
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(search_query, download=False)
                    if 'entries' in result:
                        video_info = result['entries'][0]
                        video_url = video_info['webpage_url']
                        video_id = video_info['id']
                        
                        # Embed YouTube video
                        st.markdown("### 🎥 Watch on YouTube")
                        video_embed_url = f"https://www.youtube.com/embed/{video_id}"
                        components.iframe(video_embed_url, height=315)
