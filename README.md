# Dashboard-app
Provides a dynamic dashboard for chatbot performance analysis ranging from sentiment analysis to intent

# Overview

This Streamlit dashboard allows users to visualize and monitor sentiment and intent classification results in real-time. It integrates a fine-tuned BERT-based model, which can be obtained using the python package: chatbot_analysis, to evaluate text input and display model predictions, performance metrics, and insightful trends — ideal for analyzing chatbot conversations or user feedback.

# To obtain the appropriate data format
Call the conversion function from the python package to convert from JSON to csv: https://pypi.org/project/chatbot-analysis/

# Steps to run the file 
1. Configure your API_KEY
2. python -m venv venv
3. source venv/bin/activate  # On Windows: venv\Scripts\activate
3. pip install -r requirements.txt
4. streamlit run app.py


