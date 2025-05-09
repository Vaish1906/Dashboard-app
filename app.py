import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Chatbot Analysis", page_icon=":bar_chart:",layout="wide")

#st.title(" :bar_chart: Monitoring Chatbot Sentiment Analysis")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

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
import streamlit as st
import pandas as pd
import os

st.title("File Uploader or Path Input")

# File upload or path input
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])
csv_path = st.text_input("📂 Or enter the path to a CSV/XLSX file for real-time monitoring:")

df=None
df1 = None  # Initialize dataframe
condition=True
if fl is not None or csv_path:
    try:
        if fl is not None:
            st.write(f"Uploaded File: {fl.name}")
            if fl.name.endswith(".csv") or fl.name.endswith(".txt"):
                df1= pd.read_csv(fl, encoding="ISO-8859-1")
            elif fl.name.endswith(".xlsx") or fl.name.endswith(".xls"):
                df1 = pd.read_excel(fl, engine="openpyxl")
        elif csv_path:
            if os.path.exists(csv_path):
                st.write(f"Loading from path: {csv_path}")
                if csv_path.endswith(".csv") or csv_path.endswith(".txt"):
                    df1 = pd.read_csv(csv_path, encoding="ISO-8859-1")
            else:
                st.error("File path does not exist. Please enter a valid path.")
                condition=False

        if df1 is not None:
            st.success("File Loaded Successfully!")
            st.dataframe(df1.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    columns = [
    "ï»¿_id", "Person", "stime_text", "stime_timestamp", "last_interact_text",
    "last_interact_timestamp", "llm_deployment_name", "llm_model_name", "vectorstore_index",
    "overall_cost", "overall_tokens", "role", "content", "recorded_on_text",
    "recorded_on_timestamp", "token_cost", "tokens", "user_sentiment",
    "query_intent", "conversation_id", "previous_query_intent", "overall_chat",
    "chat_sentiment", "chatbot_response_time", "overall_conversation_time"
    ]
    df = pd.DataFrame(columns)
    df1= pd.DataFrame(columns)
    condition=False
df=df1
# If "_id" exists in the dataframe columns, rename it
if "_id" in df.columns:
    df.rename(columns={"_id": "ï»¿_id"}, inplace=True)


col1, col2 = st.columns((2))
if condition:
# Convert 'recorded_on_text' to datetime (safe)
    df["recorded_on_text"] = pd.to_datetime(df["recorded_on_text"], errors="coerce")

    # Create 'conversation_id' if not already there
    df["conversation_id"] = pd.factorize(df["ï»¿_id"])[0] + 1

    # Initialize chatbot_response_time column
    df["chatbot_response_time"] = None

    # Calculate chatbot response time (loop version)
    for i in range(len(df) - 1):
        if (
            df.loc[i, "role"] == "user" and
            df.loc[i, "conversation_id"] == df.loc[i + 1, "conversation_id"]
        ):
            time_diff = (
                df.loc[i + 1, "recorded_on_text"] - df.loc[i, "recorded_on_text"]
            ).total_seconds()
            df.loc[i, "chatbot_response_time"] = time_diff

    # Calculate conversation times (safe version)
    conversation_times = df.groupby("conversation_id")["recorded_on_text"].agg(
        lambda x: (x.max() - x.min()).total_seconds() if pd.notnull(x.max()) and pd.notnull(x.min()) else None
    )

    # ✅ Assign the conversation_times back into the DataFrame (override the column)
    df = df.set_index('conversation_id')  # Set conversation_id as index temporarily for alignment
    df["overall_conversation_time"] = conversation_times  # Assign / override the column
    df = df.reset_index()  # Reset index back to normal

    # Optional: If you already have 'conversation_id' as a column, and you don't want to reset the index, you can do:
    # df["overall_conversation_time"] = df["conversation_id"].map(conversation_times)

    # Assign to df1
    df1 = df

    # Convert recorded_on_timestamp to datetime
    df["Chat Date"] = pd.to_datetime(df["recorded_on_timestamp"])
    df1["Chat Date"] = pd.to_datetime(df1["recorded_on_timestamp"])
    # Get the min and max date for the date input
    startDate = df["Chat Date"].min()
    endDate = df["Chat Date"].max()

    # Filter out AI roles
    df = df[df["role"] != "ai"]

    # Sidebar filter section
    st.sidebar.header("Choose the filter(s): ")

    # Date selection now in the sidebar
    date1 = pd.to_datetime(
        st.sidebar.date_input("Start Date", startDate)
    )

    date2 = pd.to_datetime(
        st.sidebar.date_input("End Date", endDate)
    )

    # Filter the dataframe by selected dates
    df = df[(df["Chat Date"] >= date1) & (df["Chat Date"] <= date2)]
    df1 = df1[(df1["Chat Date"] >= date1) & (df1["Chat Date"] <= date2)]
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
    
        
    def include_next_ai_rows(filtered_df1, df1):
        """
        Include the immediate next row from df1 for each row in filtered_df1 
        (based on original df1 order), if the next row's role is 'ai'.
        """
        # Reset df1 index for row position control, but preserve original index
        df1 = df1.reset_index().rename(columns={'index': 'original_index'})

        # Merge filtered_df1 with df1 to get their original positions
        merged = filtered_df1.merge(df1[['original_index']], left_index=True, right_index=True, how='left')

        # Get the original positions of the filtered rows
        filtered_original_indices = merged['original_index'].tolist()

        extra_indices = []

        for idx in filtered_original_indices:
            next_idx = idx + 1  # immediate next row in df1
            if next_idx < len(df1):
                next_row = df1.loc[next_idx]
                if str(next_row['role']).lower() == 'ai':
                    extra_indices.append(next_idx)

        # Get the next AI rows from df1 (reset the index to drop original_index column)
        extra_rows = df1.loc[extra_indices].drop(columns=['original_index'])

        # Combine filtered rows and the extra AI rows
        combined_df1 = pd.concat([filtered_df1, extra_rows]).drop_duplicates().sort_index().reset_index(drop=True)

        return combined_df1



    if not level and not query_intent and not sentiment and not conversation_id:
        filtered_df = df
        filtered_df1=df1
    
    elif level and not query_intent and not sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[df["chat_sentiment"].notnull()]
            filtered_df1 = df1
        else: 
            filtered_df = df
            filtered_df1 = df1
    
    elif not level and query_intent and not sentiment and not conversation_id:
        filtered_df = df[df["query_intent"].isin(query_intent)]
        filtered_df1 = df1[df1["query_intent"].isin(query_intent)]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif not level and not query_intent and sentiment and not conversation_id:
        filtered_df = df[df["user_sentiment"].isin(sentiment)]
        filtered_df1 = df1[df1["user_sentiment"].isin(sentiment)]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif not level and not query_intent and not sentiment and conversation_id:
        filtered_df = df[df["ï»¿_id"].isin(conversation_id)]
        filtered_df1 = df1[df1["user_sentiment"].isin(sentiment)]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and query_intent and not sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) & (df["query_intent"].isin(query_intent))]
            filtered_df1 = df1[(df1["chat_sentiment"].notnull()) & (df1["query_intent"].isin(query_intent))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[df["query_intent"].isin(query_intent)]
            filtered_df1 = df1[df1["query_intent"].isin(query_intent)]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and not query_intent and sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) & (df["chat_sentiment"].isin(sentiment))]
            filtered_df1 = df1[(df1["chat_sentiment"].notnull()) & (df1["chat_sentiment"].isin(sentiment))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[(df["user_sentiment"].isin(sentiment))]
            filtered_df1 = df1[(df1["user_sentiment"].isin(sentiment))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and not query_intent and not sentiment and conversation_id:
        if level == "Chat":
            filtered_df =  df[(df["chat_sentiment"].notnull()) & (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 =  df1[(df1["chat_sentiment"].notnull()) & (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df =  df[(df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 =  df1[(df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif not level and query_intent and sentiment and not conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) & (df["user_sentiment"].isin(sentiment))]
        filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) & (df1["user_sentiment"].isin(sentiment))]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif not level and query_intent and not sentiment and conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) & (df["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) & (df1["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif not level and not query_intent and sentiment and conversation_id:
        filtered_df = df[(df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = df1[(df1["user_sentiment"].isin(sentiment)) & (df1["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and query_intent and sentiment and not conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                             (df["query_intent"].isin(query_intent)) & (df["chat_sentiment"].isin(sentiment))]
            filtered_df1 = df1[(df1["chat_sentiment"].notnull()) &
                             (df1["query_intent"].isin(query_intent)) & (df1["chat_sentiment"].isin(sentiment))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                             (df["user_sentiment"].isin(sentiment))]
            filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) &
                             (df1["user_sentiment"].isin(sentiment))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and query_intent and not sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[ (df["chat_sentiment"].notnull()) &
                             (df["query_intent"].isin(query_intent)) & (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[ (df1["chat_sentiment"].notnull()) &
                    (df1["query_intent"].isin(query_intent)) & (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                             (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) &
                             (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    elif level and not query_intent and sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                             (df["chat_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[(df1["chat_sentiment"].notnull()) &
                             (df1["chat_sentiment"].isin(sentiment)) & (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[ (df["user_sentiment"].isin(sentiment)) &
                             (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[ (df1["user_sentiment"].isin(sentiment)) &
                             (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
                             
    
    elif not level and query_intent and sentiment and conversation_id:
        filtered_df = df[(df["query_intent"].isin(query_intent)) &
                         (df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) &
                         (df1["user_sentiment"].isin(sentiment)) & (df1["ï»¿_id"].isin(conversation_id))]
        filtered_df1 = include_next_ai_rows(filtered_df1, df1)
    
    elif level and query_intent and sentiment and conversation_id:
        if level == "Chat":
            filtered_df = df[(df["chat_sentiment"].notnull()) &
                             (df["query_intent"].isin(query_intent)) & (df["chat_sentiment"].isin(sentiment)) &
                             (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[(df1["chat_sentiment"].notnull()) &
                             (df1["query_intent"].isin(query_intent)) & (df1["chat_sentiment"].isin(sentiment)) &
                             (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
        else:
            filtered_df = df[(df["query_intent"].isin(query_intent)) &
                             (df["user_sentiment"].isin(sentiment)) & (df["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = df1[(df1["query_intent"].isin(query_intent)) &
                             (df1["user_sentiment"].isin(sentiment)) & (df1["ï»¿_id"].isin(conversation_id))]
            filtered_df1 = include_next_ai_rows(filtered_df1, df1)
            
    
    # Determine which sentiment column to use based on the level
    if level=="Chat":
        sentiment_col = "chat_sentiment"
    else:
        sentiment_col = "user_sentiment"
    unique_sentiments = filtered_df[sentiment_col].unique()
    
    filtered_df['recorded_on_date'] = pd.to_datetime(filtered_df['recorded_on_timestamp']).dt.date

    # 1️⃣ Total number of conversations
    total_conversations = filtered_df['ï»¿_id'].nunique()

    # 2️⃣ Total overall tokens
    total_tokens = filtered_df['overall_cost'].sum()

    # 3️⃣ Total number of rows/messages
    total_rows = filtered_df.shape[0]

    # 4️⃣ Weekly Chatbot Usage Growth Rate
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

    # 7️⃣ Average Chat Duration (based on one row per unique conversation)
    # Make sure there’s a 'chat_duration' column. If not, you need to add or calculate it.
    unique_chat_durations = filtered_df.drop_duplicates(subset='ï»¿_id')['overall_conversation_time']
    average_chat_duration = unique_chat_durations.mean()/60 if not unique_chat_durations.empty else 0
    average_chat_duration_str = f"{average_chat_duration:.2f} mins"

    # 8️⃣ Percentage of New Users Weekly
    recent_14_users_df = filtered_df[
        filtered_df['recorded_on_date'] >= (latest_date - pd.Timedelta(days=13))
    ]

    last_7_days_users = set(
        recent_14_users_df[
            recent_14_users_df['recorded_on_date'] > (latest_date - pd.Timedelta(days=7))
        ]['Person'].unique()
    )

    previous_7_days_users = set(
        recent_14_users_df[
            recent_14_users_df['recorded_on_date'] <= (latest_date - pd.Timedelta(days=7))
        ]['Person'].unique()
    )

    new_users_this_week = last_7_days_users - previous_7_days_users
    if len(previous_7_days_users) == 0:
        new_users_percentage = "N/A"
    else:
        new_users_percentage_value = (len(new_users_this_week) / len(previous_7_days_users)) * 100
        new_users_percentage = f"{new_users_percentage_value:.2f}%"

    # 9️⃣ Interaction Rate (Average messages per conversation)
    messages_per_conversation = filtered_df.groupby('ï»¿_id').size()
    interaction_rate = messages_per_conversation.mean() if not messages_per_conversation.empty else 0
    interaction_rate_str = f"{interaction_rate:.2f} msgs/chat"

    # ---------------- DISPLAY METRICS IN ORIGINAL FORMAT ----------------
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">⏳ Avg Chat Duration</div>
                <div class="metric-value" style="color:#884EA0;">{average_chat_duration_str}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">🆕 % Weekly New Users </div>
                <div class="metric-value" style="color:#D68910;">{new_users_percentage}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-header">💬 Interaction Rate</div>
                <div class="metric-value" style="color:#16A085;">{interaction_rate_str}</div>
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
    
    

        def generate_bubble_plot(filtered_df, level):
            #st.title("🟣 User Messages Bubble Plot (Date vs Previous Query Intent)")

            # Validate required columns
            required_columns = ['recorded_on_timestamp', 'previous_query_intent']
            missing_cols = [col for col in required_columns if col not in filtered_df.columns]
            if missing_cols:
                st.error(f"The dataframe must contain columns: {missing_cols}")
                return

            # ✅ Generate selected_sentiment dynamically
            filtered_df['selected_sentiment'] = filtered_df.apply(
                lambda row: row['chat_sentiment'] if level == 'Chat' else row['user_sentiment'],
                axis=1
            )

            # Convert timestamp to datetime if not already
            filtered_df['recorded_on_timestamp'] = pd.to_datetime(filtered_df['recorded_on_timestamp'])

            # Extract date
            filtered_df['date'] = filtered_df['recorded_on_timestamp'].dt.date

            # Group by date, previous_query_intent, and sentiment to get counts
            grouped = (
                filtered_df
                .groupby(['date', 'previous_query_intent', 'selected_sentiment'])
                .size()
                .reset_index(name='message_count')
            )

            # Color mapping for sentiment
            color_map = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'yellow'
            }

            # ✅ Create bubble plot with X-Axis as Date and Y-Axis as Previous Query Intent
            fig = px.scatter(
                grouped,
                x='date',
                y='previous_query_intent',
                size='message_count',
                color='selected_sentiment',
                color_discrete_map=color_map,
                size_max=60,
                labels={
                    'date': 'Date',
                    'previous_query_intent': 'Previous Query Intent',
                    'message_count': 'Message Count',
                    'selected_sentiment': 'Sentiment'
                },
                #title="🟣 Bubble Plot: Messages by Date and Previous Query Intent"
            )

            # Customize layout
            fig.update_layout(
                xaxis=dict(title='Date'),
                yaxis=dict(title='Previous Query Intent'),
                legend=dict(title='Sentiment'),
                height=600
            )

            # ✅ Custom CSS for visible soft box around the plot section
            st.markdown(
                """
                <style>
                .bubbleplot-container {
                    border: 1px solid #d3d3d3;  /* Light grey border */
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);  /* Thinner shadow */
                    background-color: white;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    min-height: 300px;
                }
                .bubbleplot-title {
                    text-align: center;
                    font-size: 22px;
                    font-weight: 600;
                    color: #333333;
                    margin-bottom: 5px;
                    position: relative;
                }
                .bubbleplot-title::after {
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

            # ✅ Section Title (with underline)
            st.markdown(
                """
                <h3 class='bubbleplot-title'>🫧 Bubble Plot: Previous Query Intent & Sentiment with Time</h3>
                """,
                unsafe_allow_html=True
            )

            # ✅ The container box for the plot
            #st.markdown('<div class="bubbleplot-container">', unsafe_allow_html=True)

            # Show the Plotly bubble plot
            st.plotly_chart(fig, use_container_width=True)

            # Close the container
            st.markdown('</div>', unsafe_allow_html=True)
        # Example usage:
        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)
        generate_bubble_plot(filtered_df1, level)
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
        
    def display_summary(df):
        columns_to_display = ["ï»¿_id", "role", "content", "user_sentiment", "previous_query_intent", "query_intent"]

        missing_cols = [col for col in columns_to_display if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataframe: {missing_cols}")
            return

        # Custom CSS styling
        st.markdown("""
            <style>
            .summary-container {
                border: 1px solid #d3d3d3;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .summary-title {
                text-align: center;
                font-size: 22px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 5px;
                position: relative;
            }
            .summary-title::after {
                content: "";
                display: block;
                width: 60%;
                height: 2px;
                background-color: #e0e0e0;
                margin: 8px auto 0 auto;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- Summary Table ---
        st.markdown("""<h3 class='summary-title'>💬 Chat Summary Table</h3>""", unsafe_allow_html=True)

        def color_sentiment(val):
            if pd.isna(val): return '#e0e0e0'
            val = val.lower()
            return {'positive': '#b6e3b6', 'negative': '#f5b5b5', 'neutral': '#f5f5b5'}.get(val, '#e0e0e0')

        styled_df = df[columns_to_display].style.applymap(
            lambda val: f'background-color: {color_sentiment(val)}', subset=["user_sentiment"]
        )

        #st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        csv_data = df[columns_to_display].to_csv(index=False)
        st.download_button("📥 Download Summary as CSV", csv_data, file_name="chat_summary.csv", mime="text/csv")
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        # --- Detailed View ---
        st.markdown("""<h3 class='summary-title'>📂 Detailed Chat View (Grouped by Conversation ID)</h3>""", unsafe_allow_html=True)

        unique_sentiments = df["user_sentiment"].dropna().unique().tolist()
        sentiment_filter = st.multiselect("🎭 Filter Conversations by Sentiment:", options=unique_sentiments)

        grouped = df.groupby('ï»¿_id')
        if 'expand_all' not in st.session_state:
            st.session_state['expand_all'] = False

        if st.button("Expand/Collapse All Chats"):
            st.session_state['expand_all'] = not st.session_state['expand_all']

        for conversation_id, group in grouped:
            if sentiment_filter and not group["user_sentiment"].isin(sentiment_filter).any():
                continue

            with st.expander(f"📌 Conversation ID: {conversation_id}", expanded=st.session_state['expand_all']):
                for idx, row in group.iterrows():
                    role = row['role'].lower()
                    user_sentiment = row['user_sentiment']
                    background_color = color_sentiment(user_sentiment) if role == "user" else "#f0f0f0"

                    if role == "ai":
                        role_display = "🤖 AI"
                        content_block = f"""
                            <div style="background-color:{background_color}; padding:10px; margin-bottom:10px; border-left:5px solid #ccc; border-radius:5px">
                                <h4>{role_display}</h4>
                                <p><strong>Content:</strong> {row['content']}</p>
                                {f"<p><strong>Source:</strong> {row['source']}</p>" if 'source' in df.columns and pd.notna(row['source']) else ''}
                            
                        """
                    else:
                        role_display = "👤 User"
                        content_block = f"""
                            <div style="background-color:{background_color}; padding:10px; margin-bottom:10px; border-left:5px solid #ccc; border-radius:5px">
                                <h4>{role_display}</h4>
                                <p><strong>Content:</strong> {row['content']}</p>
                                <p><strong>Sentiment:</strong> {user_sentiment or 'N/A'}</p>
                                <p><strong>Previous Query Intent:</strong> {row['previous_query_intent'] or 'N/A'}</p>
                                <p><strong>Current Query Intent:</strong> {row['query_intent'] or 'N/A'}</p>
                            </div>
                        """
                    st.markdown(content_block, unsafe_allow_html=True)

# Example call
    display_summary(filtered_df1)



    
    import openai
    from openai import OpenAI
    from dotenv import load_dotenv


    # Load environment variables from .env file
    load_dotenv()

    # Retrieve your API key
    api_key = "Insert your API KEY here"
    client = openai.OpenAI(api_key = "Insert your API KEY here")

    # Optional: Raise an error if the key is missing
    if not api_key:
        st.error("OPENAI_API_KEY is not set in environment variables.")
        st.stop()

    def generate_in_depth_summary_from_df(df):
        if df.empty:
            return "No data to summarize."

        user_messages = df[df['role'].str.lower() == 'user']['content'].dropna().tolist()
        if not user_messages:
            return "No user messages found to summarize."

        combined_messages = "\n".join([f"- {msg}" for msg in user_messages[:200]])

        prompt = f"""
        You are an expert data analyst. Analyze the following user questions and generate a detailed and comprehensive report.

        The report should include:

        1. **Key Themes**: Describe the major topics and themes that emerge from the user questions. Explain why these themes are important.

        2. **Frequently Asked Questions (FAQs)**: Provide a list of common and recurring user questions. Group similar questions where appropriate and elaborate on why users might be asking them.

        3. **Common Concerns and Pain Points**: Identify specific issues, confusion points, or frustrations that users frequently express. Offer insights into why these concerns arise.

        4. **Additional Insights**: Share any other relevant observations that may be important for understanding user behavior, needs, or opportunities for enhancing support.

        Make sure the analysis is detailed, insightful, and written in a clear narrative style.

        ---

        Here are the user questions:

        {combined_messages}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"An error occurred: {e}"

    # --- Display AI Summary ---
    def display_ai_summary(df):
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.markdown("""<h3 class='summary-title'>📊 AI-Generated Detailed Analysis</h3>""", unsafe_allow_html=True)

        if st.button("Generate Detailed AI Summary"):
            with st.spinner("Generating a comprehensive report..."):
                summary = generate_in_depth_summary_from_df(df)

                
                st.markdown(summary, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.download_button(
                    label="Download Report as TXT",
                    data=summary,
                    file_name="user_query_detailed_report.txt",
                    mime="text/plain"
                )
    display_ai_summary(filtered_df1)


    
    
    with col2:

        def generate_heatmap_with_tabs(filtered_df, level):
            #st.title("📊 User Messages Heatmap")

            # Validate required columns
            if 'recorded_on_timestamp' not in filtered_df.columns:
                st.error("The dataframe must contain 'recorded_on_timestamp'.")
                return

            # ✅ Generate selected_sentiment dynamically
            filtered_df['selected_sentiment'] = filtered_df.apply(
                lambda row: row['chat_sentiment'] if level == 'Chat' else row['user_sentiment'],
                axis=1
            )

            # Convert timestamp to datetime if not already
            filtered_df['recorded_on_timestamp'] = pd.to_datetime(filtered_df['recorded_on_timestamp'])

            # Extract day name and hour
            filtered_df['day_name'] = filtered_df['recorded_on_timestamp'].dt.day_name()
            filtered_df['hour'] = filtered_df['recorded_on_timestamp'].dt.hour

            # Ensure the day_name column follows Mon-Sun order
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            filtered_df['day_name'] = pd.Categorical(filtered_df['day_name'], categories=days_order, ordered=True)

            # Create pivot tables for each category
            pivot_tables = {
                "Overall": create_pivot(filtered_df),
                "Positive": create_pivot(filtered_df[filtered_df['selected_sentiment'].str.lower() == 'positive']),
                "Negative": create_pivot(filtered_df[filtered_df['selected_sentiment'].str.lower() == 'negative']),
                "Neutral": create_pivot(filtered_df[filtered_df['selected_sentiment'].str.lower() == 'neutral']),
            }

            # ✅ Custom CSS for visible soft box and title underline
            st.markdown(
                """
                <style>
                .heatmap-container {
                    border: 1px solid #d3d3d3;
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
                    background-color: white;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    min-height: 300px;
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
                    background-color: #e0e0e0;
                    margin: 8px auto 0 auto;
                    border-radius: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # ✅ Section Header for Heatmaps
            st.markdown(
                """
                <h3 class='sentiment-title'>🗺️ Sentiment Heatmaps</h3>
                """,
                unsafe_allow_html=True
            )

            # Create tabs for each sentiment heatmap
            tabs = st.tabs([f"{sentiment}" for sentiment in pivot_tables.keys()])

            for tab, sentiment in zip(tabs, pivot_tables.keys()):
                with tab:
                    # Color map selection
                    cmap = (
                        "Blues" if sentiment == "Overall" else
                        "Greens" if sentiment == "Positive" else
                        "Reds" if sentiment == "Negative" else
                        "YlOrBr"  # Yellow for Neutral
                    )

                    # Heatmap container
                    #st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)

                    with st.expander(f"🕒 {sentiment} Sentiment Heatmap", expanded=True):
                        plot_heatmap(pivot_tables[sentiment], cmap, title=f"{sentiment} Messages (Day vs Time)")

                    st.markdown('</div>', unsafe_allow_html=True)

        def create_pivot(df):
            """
            Create a pivot table of message counts grouped by day_name and hour.
            """
            return df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)

        def plot_heatmap(pivot_table, cmap, title):
            """
            Plot a heatmap from the pivot table.
            """
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                pivot_table.T,  # Transpose for better layout
                cmap=cmap,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Message Count'}
            )
            plt.title(title, fontsize=16)
            plt.xlabel('Day of the Week')
            plt.ylabel('Hour of the Day')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(plt.gcf())
            plt.clf()

        # Example usage:
        generate_heatmap_with_tabs(filtered_df1, level)
        st.markdown("""<div class='plot-container'>""", unsafe_allow_html=True)


    # Example usage
    # generate_heatmap(filtered_df1, level='Chat') or level='Query'

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
            labels={'adjusted_row_number': 'Conversation Number', 'sentiment_score': 'Sentiment Score'},
            color_discrete_sequence=['#85C1E9']
        )
    
        # Customize the chart
        fig.update_traces(mode='lines+markers', line=dict(width=2, dash='solid'))
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                title=dict(text='Conversation Number', font=dict(size=12, color='black', family='Arial', weight='bold'))
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
