import streamlit as st
import pandas as pd
import os
import warnings
import time
from datetime import datetime
import plotly.express as px
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
warnings.filterwarnings("ignore")
import time
import threading  # ✅ Added for background file monitoring
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Chatbot Analysis", page_icon=":bar_chart:", layout="wide")

st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        border: 2px solid black; 
        padding: 3px; 
        border-radius: 10px; 
        text-align: center; 
        background-color: #E7F3F8;
        margin-top: 20px;
    ">
        <h1 style="color: black; font-size: 30px;">📊 Monitoring Chatbot Sentiment Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)
Upload="False"
# File uploader
import streamlit as st
import pandas as pd
import os
import shutil
import time
import threading

# 📂 File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])

# 📂 Path input for continuous monitoring
csv_path = st.text_input("📂 Or enter the path to a CSV/XLSX file for real-time monitoring:")

# ✅ Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "last_modified" not in st.session_state:
    st.session_state.last_modified = None
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False  # Ensure only one thread starts

# ✅ Function to check if the file has been updated
def is_file_updated(filepath):
    if os.path.exists(filepath):
        last_modified_time = os.path.getmtime(filepath)
        if st.session_state.last_modified is None or last_modified_time > st.session_state.last_modified:
            st.session_state.last_modified = last_modified_time
            return True
    return False

# ✅ Function to load the data into session state
def load_data():
    if fl is not None:
        st.write(f"✅ Uploaded File: {fl.name}")
        try:
            if fl.name.endswith(".csv") or fl.name.endswith(".txt"):
                st.session_state.df = pd.read_csv(fl, encoding="ISO-8859-1")
            elif fl.name.endswith(".xlsx") or fl.name.endswith(".xls"):
                st.session_state.df = pd.read_excel(fl, engine="openpyxl")
        except Exception as e:
            st.error(f"🚨 Error loading file: {e}")
            st.session_state.df = None
    elif csv_path:
        if os.path.exists(csv_path):
            st.session_state.last_modified = os.path.getmtime(csv_path)
            st.write(f"✅ Loaded data from path: {csv_path}")
            try:
                temp_path = "temp_data.xlsx"
                shutil.copy(csv_path, temp_path)  # Copy file to avoid locking issues
                
                if csv_path.endswith(".csv") or csv_path.endswith(".txt"):
                    st.session_state.df = pd.read_csv(temp_path, encoding="ISO-8859-1")
                elif csv_path.endswith(".xlsx") or csv_path.endswith(".xls"):
                    st.session_state.df = pd.read_excel(temp_path, engine="openpyxl")
            except Exception as e:
                st.error(f"🚨 Error reading file: {e}")
                st.session_state.df = None
        else:
            st.error("🚨 The provided file path does not exist. Please check and try again.")
            st.session_state.df = None
    else:
        return False  # ❌ No file uploaded or path provided
    
    return True  # ✅ File successfully loa
# ✅ Load initial data
load_data()

# 📋 Display DataFrame
if st.session_state.df is not None:
    st.write("📋 Preview of Data:")
    st.dataframe(st.session_state.df.head())

# ✅ Background function to check for file updates every hour
def monitor_file():
    while True:
        time.sleep(60)  # ✅ Wait for 1 hour
        if csv_path and is_file_updated(csv_path):
            st.write("🔄 File updated! Reloading data...")
            try:
                temp_path = "temp_data.xlsx"
                shutil.copy(csv_path, temp_path)  # Copy before reading
                
                if csv_path.endswith(".csv") or csv_path.endswith(".txt"):
                    st.session_state.df = pd.read_csv(temp_path, encoding="ISO-8859-1")
                elif csv_path.endswith(".xlsx") or csv_path.endswith(".xls"):
                    st.session_state.df = pd.read_excel(temp_path, engine="openpyxl")

                st.experimental_rerun()  # ✅ Refresh Streamlit UI
            except Exception as e:
                st.error(f"🚨 Error reading updated file: {e}")

# ✅ Start monitoring in a separate thread (only once)
if csv_path and not st.session_state.monitoring:
    st.session_state.monitoring = True
    thread = threading.Thread(target=monitor_file, daemon=True)
    thread.start()
    #st.write("✅ Monitoring started in the background. The app will check for updates every hour.")

if load_data():
    col1, col2 = st.columns((2))
    df = st.session_state.df
    if "_id" in df.columns:
        df = df.rename(columns={"_id": "ï»¿_id"})
    df["Chat Date"] = pd.to_datetime(df["recorded_on_timestamp"])

    # Getting the min and max date 
    startDate = pd.to_datetime(df["Chat Date"]).min()
    endDate = pd.to_datetime(df["Chat Date"]).max()

    df = df[df["role"] != "ai"]
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["Chat Date"] >= date1) & (df["Chat Date"] <= date2)].copy()

    st.sidebar.header("Choose the filter(s): ")


    level = st.sidebar.selectbox(
        "Pick the level for sentiment analysis", 
        options=["Response","Chat"]
    )
    if level=="Chat":
        df2 = df[df["chat_sentiment"].notnull()]  # Keep only rows where "chat_sentiment" is not null   
    else:
        df2 = df.copy()  # Keep the entire dataframe if "Response" is selected

    sentiment = st.sidebar.multiselect(
        "Pick the type of sentiment", df["user_sentiment"].unique()  # Optional: Set a default value
    )
    if not sentiment :
        df3=df2.copy() # Keep only rows where "chat_sentiment" is not null   
    elif "Chat" in level:
        df3 = df2[df2["chat_sentiment"].isin(sentiment)]
    else:
        df3 = df2[df2["user_sentiment"].isin(sentiment)]

    query_intent = st.sidebar.multiselect("Pick the query intent", df["query_intent"].unique(), disabled=(level == "Chat") )
    if not query_intent:
        df4 = df3.copy()
    else:
        df4 = df3[df3["query_intent"].isin(query_intent)]

    conversation_id = st.sidebar.multiselect("Pick the conversation id", df["ï»¿_id"].unique())
    if not conversation_id:
        df5 = df4.copy()
    else:
        df5 = df4[df4["ï»¿_id"].isin(conversation_id)]


    # Assuming df is your DataFrame and the filters are defined (could be None or non-empty values)

    if not level and not query_intent and not sentiment and not conversation_id:
        filtered_df = df

    elif level and not query_intent and not sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[df["chat_sentiment"].notnull()]
        else:
            filtered_df = df

    elif not level and query_intent and not sentiment and not conversation_id:
        filtered_df = df[df["query_intent"].isin(query_intent)]

    elif not level and not query_intent and sentiment and not conversation_id:
        filtered_df = df[df["user_sentiment"].isin(sentiment)]

    elif not level and not query_intent and not sentiment and conversation_id:
        filtered_df = df[df["ï»¿_id"].isin(conversation_id)]

    elif level and query_intent and not sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) & (df["query_intent"].isin(query_intent))]
        else:
            filtered_df = df[df["query_intent"].isin(query_intent)]

    elif level and not query_intent and sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) & (df["chat_sentiment"].isin(sentiment))]
        else:
            filtered_df = df[(df["user_sentiment"].isin(sentiment))]

    elif level and not query_intent and not sentiment and conversation_id:
        if level == "Chat":
            filtered_df =  df[(df["chat_sentiment"].notnull()) & (df["ï»¿_id"].isin(conversation_id))]
        else:
            filtered_df =  df[(df["ï»¿_id"].isin(conversation_id))]

    elif not level and query_intent and sentiment and not conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) & (df["user_sentiment"].isin(sentiment))]

    elif not level and query_intent and not sentiment and conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) & (df["ï»¿_id"].isin(conversation_id))]

    elif not level and not query_intent and sentiment and conversation_id:
        filtered_df = df[(df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]

    elif level and query_intent and sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                            (df["query_intent"].isin(query_intent)) & (df["chat_sentiment"].isin(sentiment))]
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                            (df["user_sentiment"].isin(sentiment))]

    elif level and query_intent and not sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[ (df["chat_sentiment"].notnull()) &
                            (df["query_intent"].isin(query_intent)) & (df["ï»¿_id"].isin(conversation_id))]
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                            (df["ï»¿_id"].isin(conversation_id))]

    elif level and not query_intent and sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                            (df["chat_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
        else:
            filtered_df = df[ (df["user_sentiment"].isin(sentiment)) &
                            (df["ï»¿_id"].isin(conversation_id))]

    elif not level and query_intent and sentiment and conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) &
                        (df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]

    elif level and query_intent and sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                            (df["query_intent"].isin(query_intent)) & (df["chat_sentiment"].isin(sentiment)) &
                            (df["ï»¿_id"].isin(conversation_id))]
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                            (df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
            
    # Determine which sentiment column to use based on the level
    if level=="Chat":
        sentiment_col = "chat_sentiment"
    else:
        sentiment_col = "user_sentiment"

    # Get unique sentiment values from the selected sentiment column
    unique_sentiments = filtered_df[sentiment_col].unique()

    # Ensure datetime conversion
    filtered_df['recorded_on_date'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.date

    # 1️⃣ Total number of conversations
    total_conversations = filtered_df['ï»¿_id'].nunique()

    # 2️⃣ Total overall tokens
    total_tokens = filtered_df['overall_cost'].sum()

    # 3️⃣ Total number of rows
    total_rows = filtered_df.shape[0]

    # 4️⃣ Weekly Chatbot Usage Growth Rate Calculation
    latest_date = filtered_df['recorded_on_date'].max()
    recent_14_days_df = filtered_df[
        filtered_df['recorded_on_date'] >= (latest_date - pd.Timedelta(days=13))
    ]

    last_7_days_convos = recent_14_days_df[
        recent_14_days_df['recorded_on_date'] > (latest_date - pd.Timedelta(days=7))
    ]['ï»¿_id'].nunique()

    previous_7_days_convos = recent_14_days_df[
        (recent_14_days_df['recorded_on_date'] <= (latest_date - pd.Timedelta(days=7)))
    ]['ï»¿_id'].nunique()

    if previous_7_days_convos == 0:
        growth_rate = "N/A"
    else:
        growth_rate_value = ((last_7_days_convos - previous_7_days_convos) / previous_7_days_convos) * 100
        growth_rate = f"{growth_rate_value:.2f}%"

    # 5️⃣ Total number of unique users
    total_users = filtered_df['Person'].nunique()

    # 6️⃣ Percentage of repeated users (users with more than one interaction)
    user_interaction_counts = filtered_df.groupby('Person')['ï»¿_id'].nunique().reset_index(name='conversation_count')
    repeated_users = user_interaction_counts[user_interaction_counts['conversation_count'] > 1].shape[0]

    if total_users == 0:
        repeated_users_percentage = "N/A"
    else:
        repeated_users_percentage_value = (repeated_users / total_users) * 100
        repeated_users_percentage = f"{repeated_users_percentage_value:.2f}%"

    # ----------- DISPLAY ALL METRICS IN ONE BOX (Fixed) -----------
    st.markdown(
        """
        <style>
        .metric-container {
            background-color: #F4F6F7;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-header {
            font-size: 20px;
            color: #2C3E50;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">💬 Total Conversations</div>
                <div class="metric-value" style="color:#117A65;">{total_conversations:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">🔤 Total Overall Tokens</div>
                <div class="metric-value" style="color:#C0392B;">{int(total_tokens):,}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">📑 Total Rows</div>
                <div class="metric-value" style="color:#B7950B;">{total_rows:,}</div>
            </div>
        """, unsafe_allow_html=True)

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">📈 Weekly Growth Rate</div>
                <div class="metric-value" style="color:#2471A3;">{growth_rate}</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">🧑‍💻 Total Unique Users</div>
                <div class="metric-value" style="color:#C0392B;">{total_users:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">🔄 Repeated Users (%)</div>
                <div class="metric-value" style="color:#2471A3;">{repeated_users_percentage}</div>
            </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
    # Custom CSS for background, centered sentiment title, and underline
    # ---------- Sentiment Analysis Word Clouds ----------

    # Custom CSS for visible soft box around the section
        st.markdown(
            """
            <style>
            .wordcloud-container {
                border: 1px solid #d3d3d3;  /* Light grey border */
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);  /* Thinner shadow */
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 200px; /* Standardized minimum height for consistency */
            }
            .sentiment-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .sentiment-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #e0e0e0; /* Light grey underline for consistency */
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Main header with cloud icon
        st.markdown(
            """
            <h3 class='sentiment-title'>☁️ Sentiment Analysis Word Clouds</h3>
            """,
            unsafe_allow_html=True
        )

        # Create tabs for each sentiment
        tabs = st.tabs([f"{sentiment.capitalize()}" for sentiment in unique_sentiments])

        # Loop through each sentiment and display word cloud in corresponding tab
        for tab, sentiment in zip(tabs, unique_sentiments):
            with tab:
                # Filter rows for the current sentiment based on the level
                if "Chat" in level:
                    sentiment_texts = filtered_df[filtered_df[sentiment_col] == sentiment]["overall_chat"]
                else:
                    sentiment_texts = filtered_df[filtered_df[sentiment_col] == sentiment]["content"]

                # Combine all text into a single string
                text = " ".join(sentiment_texts.dropna().astype(str).tolist())

                # Generate the word cloud if text is not empty
                if text.strip():
                    with st.expander(f"🌟 {sentiment.capitalize()} Word Cloud", expanded=True):
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                        # Create a matplotlib figure
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')

                        # Word cloud container with centered sentiment title and underline
                        #st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)

                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(f"💡 No text available for **{sentiment.capitalize()}** sentiment.")

        # ---------- Sentiment Distribution Donut Chart ----------

        # Custom CSS for visible soft box around the plot with underline under the title
        st.markdown(
            """
            <style>
            .plot-container {
                border: 2px solid #f0f0f5;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                background-color: #f9f9fc;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 3px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)
        st.markdown(
            """
            <h3 class='plot-title'>💬 Sentiment Distribution</h3>
            """,
            unsafe_allow_html=True
        )

        # Select sentiment column based on level
        filtered_df['selected_sentiment'] = filtered_df.apply(
            lambda row: row['user_sentiment'] if level == 'Response' else row['chat_sentiment'],
            axis=1
        )

        # Count sentiment occurrences
        sentiment_counts = filtered_df['selected_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Create the compact donut chart without legend
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2,
            width=500,  # Standardized width
            height=400  # Standardized height
        )

        # Update trace to display sentiment labels outside
        fig.update_traces(
            text=sentiment_counts['Sentiment'],
            textposition='outside',
            textfont_size=12,
            hovertemplate='Sentiment: %{label}<br>Count: %{value}<extra></extra>'
        )

        # Chart layout updates without legend
        fig.update_layout(
            showlegend=False,  # Legend removed
            hovermode="closest"
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    # ---------- Query Intent Distribution Bar Chart ----------

    # Custom CSS for visible soft box around the plot with underline under the title
    # ---------- Query Intent Distribution Bar Chart ----------

    # Custom CSS for visible soft box around the plot with underline under the title
        st.markdown(
            """
            <style>
            .plot-container {
                border: 2px solid #f0f0f5;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                background-color: #f9f9fc;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 3px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if level == "Response":
            st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class='plot-title'>🎯 Query Intent Distribution</h3>
                """,
                unsafe_allow_html=True
            )

            # Filter data where level == "Response"
            response_df = filtered_df

            if not response_df.empty:
                # Count occurrences of each query_intent
                query_intent_counts = response_df['query_intent'].value_counts().reset_index()
                query_intent_counts.columns = ['Query Intent', 'Count']

                # Create the bar chart using Plotly
                fig = px.bar(
                    query_intent_counts,
                    x='Query Intent',
                    y='Count',
                    text='Count',
                    color='Query Intent',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                # Customize the bar chart with bold black axis titles
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(
                    xaxis_title=dict(text='Query Intent', font=dict(color='black', size=14, family='Arial', weight='bold')),
                    yaxis_title=dict(text='Count', font=dict(color='black', size=14, family='Arial', weight='bold')),
                    xaxis_tickangle=-45,
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    showlegend=False,
                    height=500,
                    width=900,
                    margin=dict(l=100, r=20, t=50, b=80),  # Adjusted for complete y-axis visibility
                    yaxis=dict(automargin=True, title_font=dict(color='black', size=14, family='Arial', weight='bold')),
                    xaxis=dict(title_font=dict(color='black', size=14, family='Arial', weight='bold'))
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No data available for the selected filter: Response.")
            st.markdown("""</div>""", unsafe_allow_html=True)
        else:
            st.info("ℹ️ The Query Intent Distribution chart is only available when the selected level is **Response**.")
        # ---------- Sentiment Score Distribution by Query Intent ----------

        # Custom CSS for visible soft box around the plot with underline under the title
        st.markdown(
            """
            <style>
            .plot-container {
                border: 2px solid #f0f0f5;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                background-color: #f9f9fc;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 3px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if level == "Response":
            st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class='plot-title'>📊 Sentiment Distribution by Query Intent</h3>
                """,
                unsafe_allow_html=True
            )

            # Filter data where level == "Response"
            response_df = filtered_df

            if not response_df.empty:
                # Map sentiments to scores
                sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
                response_df['sentiment_score'] = response_df['user_sentiment'].map(sentiment_mapping)

                # Group data by sentiment_score and query_intent
                sentiment_intent_counts = response_df.groupby(['sentiment_score', 'previous_query_intent']).size().reset_index(name='Count')

                # Create the stacked bar chart using Plotly
                fig = px.bar(
                    sentiment_intent_counts,
                    x='sentiment_score',
                    y='Count',
                    color='previous_query_intent',
                    barmode='stack',
                    text='Count',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                # Customize the chart with bold black axis titles and clickable legend
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(
                    xaxis_title=dict(text='Sentiment Score', font=dict(color='black', size=14, family='Arial', weight='bold')),
                    yaxis_title=dict(text='Count', font=dict(color='black', size=14, family='Arial', weight='bold')),
                    xaxis=dict(
                        tickvals=[-1, 0, 1],
                        ticktext=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
                        title_font=dict(color='black', size=14, family='Arial', weight='bold')
                    ),
                    yaxis=dict(
                        automargin=True,
                        title_font=dict(color='black', size=14, family='Arial', weight='bold')
                    ),
                    legend_title='Query Intent',
                    legend=dict(
                        font=dict(size=7),
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="right",
                        x=1,
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        bordercolor='black',
                        borderwidth=1
                    ),
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    height=500,
                    width=900,
                    margin=dict(l=100, r=20, t=50, b=80)  # Adjusted for complete axis visibility
                )

                # Enable click-to-hide/show functionality (default in Plotly)
                fig.update_layout(legend_itemclick="toggle", legend_itemdoubleclick="toggleothers")

                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No data available for the selected filter: Response.")
            st.markdown("""</div>""", unsafe_allow_html=True)
        else:
            st.info("ℹ️ The Stacked Sentiment Score chart is only available when the selected level is **Response**.")


    # ---------- Negative Sentiment Conversations ----------

    # Custom CSS for visible soft box around the section and each conversation block
        st.markdown(
            """
            <style>
            .content-container {
                border: 1px solid #d3d3d3;  /* Light grey border */
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);  /* Thinner shadow */
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                
            }
            .content-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .content-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #e0e0e0; /* Light grey underline for consistency */
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            .content-box-wrapper {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
                background-color: #ffffff;
            }
            .content-box {
                padding: 10px;
                border-radius: 10px;
                background-color: #f8d7da;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""<div class='content-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='content-title'>❗ Negative Sentiment Conversations</h3>
            """,
            unsafe_allow_html=True
        )

        # Dynamically filter negative sentiment based on level
        if level == "Chat":
            negative_sentiment_df = filtered_df[filtered_df['chat_sentiment'].str.lower() == 'negative']
        else:
            negative_sentiment_df = filtered_df[filtered_df['user_sentiment'].str.lower() == 'negative']

        if not negative_sentiment_df.empty:
            # Get unique negative conversation IDs
            negative_conversations = negative_sentiment_df['ï»¿_id'].unique()

            # Dropdown menu to select conversation ID
            selected_conversation = st.selectbox(
                "🔍 Select a Conversation ID with Negative Sentiment:",
                negative_conversations
            )

            # Display content for the selected conversation
            selected_convo_df = negative_sentiment_df[negative_sentiment_df['ï»¿_id'] == selected_conversation]

            for idx, row in selected_convo_df.iterrows():
                st.markdown(f"""
                    <div class='content-box-wrapper'>
                        <div class='content-box'>
                            <b>Timestamp:</b> {row['recorded_on_timestamp']}<br>
                            <b>Content:</b> {row['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No conversations found with negative sentiment based on the specified level logic.")

        st.markdown("""</div>""", unsafe_allow_html=True)


        # ---------- Negative Sentiment Analysis by Person ----------

        # Custom CSS for visible soft box around the section
    # ---------- Negative Sentiment Analysis by Person ----------

    # Custom CSS for visible soft box around the section
        st.markdown(
            """
            <style>
            .person-analysis-container {
                border: 1px solid #d3d3d3;  /* Light grey border */
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);  /* Thinner shadow */
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                max-height: 15px; /* Standardized minimum height for consistency */
            }
            .person-analysis-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .person-analysis-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #e0e0e0; /* Light grey underline for consistency */
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""<div class='person-analysis-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='person-analysis-title'>👤 Negative Sentiment Analysis by Person</h3>
            """,
            unsafe_allow_html=True
        )

        if not negative_sentiment_df.empty:
            # Count unique people who expressed negative sentiment
            unique_people = negative_sentiment_df['Person'].nunique()
            st.success(f"🧑‍🤝‍🧑 **People who expressed negative sentiments:** {unique_people}")

            # Group by person and count distinct conversations where negative sentiment was expressed
            person_convo_counts = (
                negative_sentiment_df.groupby('Person')['ï»¿_id']
                .nunique()
                .reset_index(name='distinct_negative_conversations')
            )

            # Filter for people who expressed negative sentiment in more than one conversation
            multiple_negative_convos = person_convo_counts[person_convo_counts['distinct_negative_conversations'] > 1]

            # Display the count of such individuals
            multi_count = multiple_negative_convos.shape[0]
            st.info(f"🔄 **Individuals who expressed negative sentiments more than once:** {multi_count}")

            if multi_count > 0:
                # Show details of these individuals using full column width
                st.dataframe(multiple_negative_convos, use_container_width=True)
            else:
                st.info("✅ **No individual expressed negative sentiment in more than one distinct conversation.**")
        else:
            st.info("ℹ️ **No negative sentiments found based on the specified level logic.**")

        st.markdown("""</div>""", unsafe_allow_html=True)




    with col2:
    # ---------- Continuous Sentiment Line Plot (Standardized) ----------

    # Sort by conversation_id and timestamp
    # ---------- Continuous Sentiment Line Plot (Updated with Plotly Express, Standardized, and No Extra Top Box) ----------

        # Sort by conversation_id and timestamp
        filtered_df.sort_values(by=['ï»¿_id', 'recorded_on_timestamp'], inplace=True)
        filtered_df.reset_index(drop=True, inplace=True)

        # Add a row number column
        filtered_df['row_number'] = filtered_df.index + 1

        # Map sentiments
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        filtered_df['sentiment_score'] = filtered_df[sentiment_col].map(sentiment_mapping)

        # Add padding to row_number when switching conversations
        padding = 1
        adjusted_row_number = []
        additional_padding = 0
        prev_conv_id = filtered_df.loc[0, 'ï»¿_id']

        for idx, row in filtered_df.iterrows():
            curr_conv_id = row['ï»¿_id']
            if idx > 0 and curr_conv_id != prev_conv_id and level == 'Response':
                additional_padding += padding
            adjusted_row_number.append(row['row_number'] + additional_padding)
            prev_conv_id = curr_conv_id

        filtered_df['adjusted_row_number'] = adjusted_row_number

        # Create the continuous line plot using Plotly Express (px.line)
        fig = px.line(
            filtered_df,
            x='adjusted_row_number',
            y='sentiment_score',
            markers=True,
            labels={'adjusted_row_number': 'Row Number', 'sentiment_score': 'Sentiment Score'},
            color_discrete_sequence=['#85C1E9']
        )

        # Customize the chart
        fig.update_traces(mode='lines+markers', line=dict(width=2, dash='solid'))
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                title=dict(text='Row Number', font=dict(size=12, color='black', family='Arial', weight='bold'))
            ),
            yaxis=dict(
                tickvals=[-1, 0, 1],
                ticktext=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
                title=dict(text='Sentiment Score', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            height=550,
            width=900,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add orange vertical dotted lines for conversation separation
        shapes = []
        prev_conv_id = filtered_df.loc[0, 'ï»¿_id']
        for idx, row in filtered_df.iterrows():
            curr_conv_id = row['ï»¿_id']
            if idx > 0 and curr_conv_id != prev_conv_id and level == 'Response':
                shapes.append(
                    dict(
                        type="line",
                        x0=row['adjusted_row_number'],
                        y0=-1,
                        x1=row['adjusted_row_number'],
                        y1=1,
                        line=dict(color="orange", width=1, dash="dot")
                    )
                )
            prev_conv_id = curr_conv_id

        fig.update_layout(shapes=shapes)

        # Display the chart in Streamlit (with standardized box styling and underlined title)
        st.markdown(
            """
            <style>
            .line-plot-container {
                border: 2px solid #f0f0f5;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
                background-color: white;
                height: 600px;
            }
            .line-plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .line-plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 3px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        #st.markdown("""<div class='line-plot-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='line-plot-title'>⏰ Continuous Sentiment Line Plot</h3>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)



        # ---------- Average Daily Sentiment Over Time (With Standardized Box Styling) ----------

        # Convert timestamp to datetime and extract date
        filtered_df['recorded_on_date'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.date

        # Select sentiment column based on level
        filtered_df['selected_sentiment'] = filtered_df.apply(
            lambda row: row['chat_sentiment'] if level == 'Chat' else row['user_sentiment'],
            axis=1
        )

        # Map sentiments to scores
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        filtered_df['sentiment_score'] = filtered_df['selected_sentiment'].map(sentiment_mapping)

        # Group by date and calculate average sentiment
        daily_avg_sentiment = filtered_df.groupby('recorded_on_date')['sentiment_score'].mean().reset_index()

        # Create the line chart using Plotly
        fig = px.line(
            daily_avg_sentiment,
            x='recorded_on_date',
            y='sentiment_score',
            markers=True,
            labels={'recorded_on_date': 'Date', 'sentiment_score': 'Average Sentiment Score'},
            color_discrete_sequence=['#1f77b4']
        )

        # Customize the chart
        fig.update_traces(mode='lines+markers', line=dict(width=2))
        fig.update_layout(
            xaxis=dict(
                title=dict(text='Date', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            yaxis=dict(
                tickvals=[-1, 0, 1],
                ticktext=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
                title=dict(text='Average Sentiment Score', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            height=550,
            width=900,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Display the chart in Streamlit (with updated plot container styling)
        st.markdown(
            """
            <style>
            .plot-container {
                border: 1px solid #d3d3d3;  /* Light grey border */
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);  /* Thinner shadow */
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 3px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='plot-title'>📈 Average Daily Sentiment Over Time</h3>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)



        # ---------- Total Token Cost & Number of Conversations Per Day (With Standardized Box Styling) ----------

        # Convert timestamp to datetime and extract date
        filtered_df['recorded_on_date'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.date

        # Calculate total cost per day
        daily_total_cost = filtered_df.groupby('recorded_on_date')['overall_cost'].sum().reset_index()
        # Calculate number of distinct conversations per day
        daily_conversations = filtered_df.groupby('recorded_on_date')['ï»¿_id'].nunique().reset_index()
        daily_conversations.rename(columns={'ï»¿_id': 'num_conversations'}, inplace=True)

        # Merge both datasets on the date
        daily_summary = pd.merge(daily_total_cost, daily_conversations, on='recorded_on_date')

        # Create the combined line chart using Plotly Graph Objects
        fig = go.Figure()

        # Line for Total Token Cost
        fig.add_trace(
            go.Scatter(
                x=daily_summary['recorded_on_date'],
                y=daily_summary['overall_cost'],
                mode='lines+markers',
                name='Total Token Cost',
                line=dict(color='#FF7F0E', width=2),
                yaxis='y1'
            )
        )

        # Line for Number of Conversations
        fig.add_trace(
            go.Scatter(
                x=daily_summary['recorded_on_date'],
                y=daily_summary['num_conversations'],
                mode='lines+markers',
                name='Number of Conversations',
                line=dict(color='#1f77b4', width=2, dash='dot'),
                yaxis='y2'
            )
        )

        # Customize the chart layout with dual y-axes
        fig.update_layout(
            xaxis=dict(title='Date', showgrid=True),
            yaxis=dict(
                title='Total Token Cost',
                showgrid=True,
                zeroline=True
            ),
            yaxis2=dict(
                title='Number of Conversations',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                font=dict(size=10),
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1
            ),
            height=550,
            width=900,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Display the chart in Streamlit (with updated plot container styling)
        st.markdown(
            """
            <style>
            .plot-container {
                border: 1px solid #e0e0e0;  /* Light grey border */
                border-radius: 10px;        /* Slightly thinner rounded corners */
                padding: 15px;              /* Reduced padding */
                margin-bottom: 20px;       /* Reduced margin */
                box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.03);  /* Softer shadow */
                background-color: white;   /* White background */
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 2px;              /* Thinner underline */
                background-color: #d3d3d3; /* Lighter grey underline */
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='plot-title'>💬 Total Token Cost & Number of Conversations Per Day</h3>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)

        # ---------- Overall Chatbot Usage Over Time (With Standardized Box Styling) ----------

        # Convert timestamp to datetime and extract date
        filtered_df['recorded_on_date'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.date

        # Drop duplicates based on 'conversation_id' per day to consider only unique conversations
        unique_conversations_per_day = filtered_df.drop_duplicates(subset=['recorded_on_date', 'ï»¿_id'])

        # Group by date and sum the 'overall_conversation_time' per day
        daily_chatbot_usage = unique_conversations_per_day.groupby('recorded_on_date')['overall_conversation_time'].sum().reset_index()
        daily_chatbot_usage['overall_conversation_time'] = daily_chatbot_usage["overall_conversation_time"] / 360

        # Create the line chart using Plotly
        fig = px.line(
            daily_chatbot_usage,
            x='recorded_on_date',
            y='overall_conversation_time',
            markers=True,
            labels={'recorded_on_date': 'Date', 'overall_conversation_time': 'Total Chatbot Usage (Hours)'},
            color_discrete_sequence=['#2ca02c']
        )

        # Customize the chart
        fig.update_traces(mode='lines+markers', line=dict(width=2))
        fig.update_layout(
            xaxis=dict(
                title=dict(text='Date', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text='Total Chatbot Usage (Hours)', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            height=550,
            width=900,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Display the chart in Streamlit (with updated plot container styling)
        st.markdown(
            """
            <style>
            .plot-container {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.03);
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: """""";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #d3d3d3;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='plot-title'>🤖 Overall Chatbot Usage Over Time</h3>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)

    # ---------- Most Active Hours of the Day (AM/PM Format) ----------

        st.markdown(
            """
            <style>
            .plot-container {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.03);
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .plot-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .plot-title::after {
                content: """""";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #d3d3d3;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)

        st.markdown(
            """
            <h3 class='plot-title'>⏰ Most Active Hours of the Day (AM/PM Format)</h3>
            """,
            unsafe_allow_html=True
        )

        # Extract hour from the timestamp in 12-hour format with AM/PM
        filtered_df['recorded_hour'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.strftime('%I %p')

        # Ensure the order of hours for proper sorting
        hour_order = [f"{hour:02d} AM" for hour in range(1, 12)] + ['12 PM'] + [f"{hour:02d} PM" for hour in range(1, 12)] + ['12 AM']

        # Count the number of conversations per hour
        hourly_activity = filtered_df.groupby('recorded_hour').size().reset_index(name='conversation_count')

        # Sort the DataFrame based on the custom hour order
        hourly_activity['recorded_hour'] = pd.Categorical(hourly_activity['recorded_hour'], categories=hour_order, ordered=True)
        hourly_activity.sort_values('recorded_hour', inplace=True)

        # Create the bar chart using Plotly
        fig = px.bar(
            hourly_activity,
            x='recorded_hour',
            y='conversation_count',
            text='conversation_count',
            labels={'recorded_hour': 'Hour of the Day', 'conversation_count': 'Number of Conversations'},
            color='conversation_count',
            color_continuous_scale='Viridis'
        )

        # Customize the chart
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            xaxis=dict(
                title=dict(text='Hour of the Day', font=dict(size=12, color='black', family='Arial', weight='bold')),
                tickmode='array',
                tickvals=hour_order,
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text='Number of Conversations', font=dict(size=12, color='black', family='Arial', weight='bold')),
                showgrid=True
            ),
            coloraxis_showscale=False,
            height=550,
            width=900,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""</div>""", unsafe_allow_html=True)
