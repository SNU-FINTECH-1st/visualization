import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Load and preprocess data
@st.cache_data
def load_data():
    df2 = pd.read_csv("1jo/songs_normalize.csv")
    df2 = df2.dropna(subset=['genre'])
    df2['genre'] = df2['genre'].str.split(',')
    df2 = df2.explode('genre').reset_index(drop=True)
    df2['genre'] = df2['genre'].str.strip().str.title()
    return df2[df2['genre'] != 'Set()']

# Streamlit Layout
def setup_layout():
    st.title("Music Genre Visualizations")

def genre_year_filters(df2):
    st.write("### Filters")
    
    # Create columns for filters
    col1, col2 = st.columns(2)

    # Get unique genres
    genres = df2['genre'].unique().tolist()
    num_columns = 4  # Number of columns to display genres

    # Adjust margin to align "Select All Genres" with other checkboxes
    with col1:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # Adjust the margin as needed
        all_genres = st.checkbox("Select All Genres", value=True)

    selected_genres = []

    # Display genres in multiple columns
    genre_cols = st.columns(num_columns)
    
    for i, genre in enumerate(genres):
        col = genre_cols[i % num_columns]
        with col:
            if all_genres:
                checked = True
            else:
                checked = False

            if st.checkbox(genre, value=checked, key=genre):
                selected_genres.append(genre)

    # Year range slider in col2
    with col2:
        min_year, max_year = max(2000, int(df2['year'].min())), min(2019, int(df2['year'].max()))
        selected_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    return selected_genres, selected_years

def filter_data(df2, selected_genres, selected_years):
    filtered_df2 = df2[
        (df2['genre'].isin(selected_genres)) & 
        (df2['year'].between(*selected_years))
    ]
    
    # If the dataframe is empty, display a humorous message
    if filtered_df2.empty:
        st.warning("🚨 Oops! It looks like you didn't select any genres.")
        st.image("https://media.giphy.com/media/13FysNaqRo3UIM/giphy.gif", use_column_width=True)
        return None
    
    return filtered_df2

# Visualization functions
def generate_streamgraph(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Stream Graph: Genre Popularity Over Time")
    genre_year_counts = filtered_df2.groupby(['year', 'genre']).size().unstack(fill_value=0)
    data_for_streamgraph = genre_year_counts.T

    plt.figure(figsize=(14, 8))
    plt.stackplot(data_for_streamgraph.columns, data_for_streamgraph, labels=data_for_streamgraph.index)
    plt.title('Stream Graph of Genre Popularity Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Songs', fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def generate_radar_chart(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Radar Chart: Genre Influence on Song Characteristics")
    characteristics = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    radar_data = filtered_df2.groupby('genre')[characteristics].mean().reset_index()

    fig = go.Figure()

    for i in range(len(radar_data)):
        fig.add_trace(go.Scatterpolar(
            r=radar_data.loc[i, characteristics].values,
            theta=characteristics,
            fill='toself',
            name=radar_data['genre'].iloc[i]  # Correctly access the genre name here
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Radar Chart of Genre Influence on Song Characteristics"
    )

    st.plotly_chart(fig)

def generate_treemap(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Tree Map: Genre Distribution in Top Songs")
    genre_counts = filtered_df2['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    fig = px.treemap(genre_counts, path=['genre'], values='count', title="Tree Map of Genre Distribution", color='count', color_continuous_scale='Viridis')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig)

def generate_bubble_chart(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Bubble Chart: Correlation between Genre and Popularity")
    genre_popularity = filtered_df2.groupby('genre').agg({'popularity': 'mean', 'genre': 'size'})
    genre_popularity.columns = ['average_popularity', 'count']
    genre_popularity = genre_popularity.reset_index()
    fig = px.scatter(genre_popularity, x='average_popularity', y='genre', size='count', color='genre', hover_name='genre', title="Bubble Chart of Genre Popularity", size_max=60)
    fig.update_layout(xaxis_title='Average Popularity', yaxis_title='Genre', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), showlegend=False)
    st.plotly_chart(fig)

def generate_heatmap(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Heatmap: Audio Features Over Time Across Genres")
    
    # Dropdown menu within the main content area
    characteristics = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    selected_characteristic = st.selectbox("Select Characteristic to Visualize", options=characteristics, index=0)
    
    heatmap_data = filtered_df2.groupby(['year', 'genre'])[selected_characteristic].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data.T, 
        labels=dict(x="Year", y="Genre", color=f"Average {selected_characteristic.capitalize()}"),
        title=f"Heatmap of {selected_characteristic.capitalize()} Over Time"
    )
    
    st.plotly_chart(fig)

def generate_network_graph(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Network Graph: Genre Co-occurrence Network")
    network_data = filtered_df2.dropna(subset=['genre'])
    
    # Split genres and create genre pairs
    genre_pairs = network_data.groupby('song')['genre'].apply(lambda x: list(pd.unique(x)))
    edges = [(g1, g2) for pairs in genre_pairs for i, g1 in enumerate(pairs) for g2 in pairs[i+1:]]
    
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Check if the graph has nodes
    if len(G.nodes) == 0:
        st.warning("No genre pairs available to create a network graph.")
        return
    
    pos = nx.spring_layout(G, k=0.3, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y = zip(*[pos[node] for node in G.nodes()])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=10, color='skyblue', line_width=2),
        text=list(G.nodes()),
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Network Graph of Genre Co-occurrence",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40)
    )

    st.plotly_chart(fig)

def generate_sunburst_chart(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Sunburst Chart: Genre Hierarchy")
    genre_counts = filtered_df2['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    genre_counts['parent'] = 'Music'
    
    fig = px.sunburst(
        genre_counts,
        names='genre',
        parents='parent',
        values='count',
        title="Sunburst Chart of Genre Distribution"
    )
    st.plotly_chart(fig)

def generate_tempo_histogram(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Histogram of Average Tempo by Genre")
    
    tempo_data = filtered_df2.groupby('genre')['tempo'].mean().reset_index()
    
    fig = px.histogram(tempo_data, x='genre', y='tempo', nbins=20, title="Distribution of Average Tempo by Genre")
    
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Average Tempo',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        bargap=0.1
    )
    
    st.plotly_chart(fig)

def generate_popularity_trend(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Popularity Trend Analysis by Genre")
    
    trend_data = filtered_df2.groupby(['year', 'genre'])['popularity'].mean().reset_index()
    
    fig = px.line(trend_data, x='year', y='popularity', color='genre', title="Trend Analysis of Popularity by Genre")
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Popularity',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig)

def generate_duration_boxplot(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Box Plot of Song Duration by Genre")
    
    filtered_df2['duration_min'] = filtered_df2['duration_ms'] / 60000  # Convert milliseconds to minutes
    
    fig = px.box(filtered_df2, x='genre', y='duration_min', title="Comparison of Song Duration by Genre")
    
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Song Duration (Minutes)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig)

def generate_stacked_bar_chart(filtered_df2):
    if filtered_df2 is None:
        return
    
    st.subheader("Stacked Bar Chart of Song Count by Genre and Year")
    
    stacked_data = filtered_df2.groupby(['year', 'genre']).size().unstack(fill_value=0)
    
    fig = px.bar(stacked_data, title="Number of Songs by Genre and Year", labels={"value": "Number of Songs", "index": "Year"}, barmode='stack')
    
    st.plotly_chart(fig)

# 기존 main 함수를 app 함수로 변경
def app():
    df2 = load_data()
    setup_layout()
    selected_genres, selected_years = genre_year_filters(df2)
    filtered_df2 = filter_data(df2, selected_genres, selected_years)

    if filtered_df2 is None:
        return
    
    st.write(f"Displaying data for {len(filtered_df2)} songs.")
    
    tab1, tab2, tab3 = st.tabs(["Time-based", "Feature Analysis", "Genre Distribution"])
    
    with tab1:
        generate_streamgraph(filtered_df2)
        generate_heatmap(filtered_df2)
    
    with tab2:
        generate_radar_chart(filtered_df2)

    with tab3:
        generate_treemap(filtered_df2)
        generate_sunburst_chart(filtered_df2)
        generate_bubble_chart(filtered_df2)
        generate_network_graph(filtered_df2)
        generate_stacked_bar_chart(filtered_df2)

# __name__에 따라 실행
if __name__ == "__main__":
    app()
