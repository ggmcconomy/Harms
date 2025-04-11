import os
import json
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import faiss
import openai

# Use Streamlit secrets for sensitive values
openai.api_key = st.secrets["OPENAI_API_KEY"]
MURAL_API_TOKEN = st.secrets["MURAL_API_TOKEN"]
MURAL_ID = st.secrets["MURAL_ID"]

# Load dataset
csv_file = 'AI-Powered_Valuation_Enriched.csv'
df = pd.read_csv(csv_file)

# Preprocess text
def preprocess_text(text):
    text = str(text).lower()
    return ''.join([c for c in text if c.isalnum() or c.isspace()])

df['processed_description'] = df['risk_description'].apply(preprocess_text)

# Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
csv_embeddings = embedder.encode(df['processed_description'].tolist(), show_progress_bar=False)

# Cluster for theme detection
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(csv_embeddings)

# Build FAISS index
dimension = csv_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(csv_embeddings)

# Streamlit app
st.title("AI-Powered Risk Feedback & Mural Integration")

params = st.experimental_get_query_params()
session_id = params.get('session', [None])[0]

# Load session
if session_id:
    try:
        with open(f"sessions/{session_id}.json", "r") as f:
            session_data = json.load(f)
        st.markdown("### Previously Generated Feedback")
        st.markdown(session_data["feedback"])
        missed_risks = session_data["top_missed_risks"]
    except FileNotFoundError:
        st.warning("Session not found.")
        st.stop()
else:
    # Human input
    st.markdown("## Input Your Risks")
    human_input = st.text_area("Paste risks below (one per line):")
    if st.button("Run Analysis"):
        human_risks = [r.strip() for r in human_input.split('\n') if r.strip()]
        human_embeddings = embedder.encode(human_risks)
        distances, indices = index.search(human_embeddings, 5)
        similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]

        human_clusters = set([r['cluster'] for group in similar_risks for r in group])
        human_types = set([r['risk_type'] for group in similar_risks for r in group])
        human_stakeholders = set([r['stakeholder'] for group in similar_risks for r in group])

        missed_clusters = set(df['cluster']) - human_clusters
        missed_types = set(df['risk_type']) - human_types
        missed_stakeholders = set(df['stakeholder']) - human_stakeholders

        high_severity_df = df[df['severity'] >= 4.0]
        missed_high_severity = high_severity_df[~high_severity_df['cluster'].isin(human_clusters)]
        top_missed = missed_high_severity.sort_values(by='combined_score', ascending=False).head(5)

        # Construct prompt
        prompt = f"""
        The user provided these risks: {', '.join(human_risks)}.
        Based on our database, they missed these critical areas:
        - Themes: {', '.join(map(str, missed_clusters))}
        - Risk Types: {', '.join(missed_types)}
        - Stakeholders: {', '.join(missed_stakeholders)}
        Top missed high-severity risks:
        {chr(10).join('- ' + r for r in top_missed['risk_description'].tolist())}
        Provide feedback to improve their risk analysis.
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI risk advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        feedback = response.choices[0].message.content
        st.markdown("### Feedback from AI")
        st.markdown(feedback)

        # Save session
        session_id = str(uuid.uuid4())
        os.makedirs("sessions", exist_ok=True)
        with open(f"sessions/{session_id}.json", "w") as f:
            json.dump({
                "feedback": feedback,
                "top_missed_risks": top_missed.to_dict(orient="records")
            }, f)
        st.success("Session saved!")
        st.markdown(f"[Reopen this session](?session={session_id})")
        missed_risks = top_missed.to_dict(orient="records")

# Post risks to Mural
if "missed_risks" in locals():
    if "posted_count" not in st.session_state:
        st.session_state.posted_count = 0

    if st.button("Post Next Missed Risk to Mural"):
        if st.session_state.posted_count < len(missed_risks):
            risk = missed_risks[st.session_state.posted_count]
            mural_url = f"https://your-streamlit-app.com/?session={session_id}"
            payload = {
                "x": 1000 + st.session_state.posted_count * 100,
                "y": 1000,
                "width": 300,
                "height": 150,
                "text": f"\ud83e\udde0 AI Risk Suggestion:\n{risk['risk_description']}\n\n[More]( {mural_url} )",
                "shape": "square",
                "color": "yellow",
                "layer": "content",
                "type": "sticky_note"
            }
            headers = {
                'Authorization': f'Bearer {MURAL_API_TOKEN}',
                'Content-Type': 'application/json'
            }
            mural_response = requests.post(
                f"https://api.mural.co/v1/workspaces/{MURAL_WORKSPACE_ID}/murals/{MURAL_BOARD_ID}/widgets",
                headers=headers,
                json=payload
            )
            if mural_response.status_code in [200, 201]:
                st.success(f"Posted: {risk['risk_description']}")
                st.session_state.posted_count += 1
            else:
                st.error(f"Failed to post to Mural: {mural_response.text}")
        else:
            st.info("All top risks have been posted.")
