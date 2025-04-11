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

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
MURAL_API_TOKEN = st.secrets["MURAL_API_TOKEN"]
MURAL_WORKSPACE_ID = st.secrets["MURAL_WORKSPACE_ID"]
MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]

# --- Load Risk Dataset ---
csv_file = 'AI-Powered_Valuation_Enriched.csv'
df = pd.read_csv(csv_file)

def preprocess_text(text):
    text = str(text).lower()
    return ''.join([c for c in text if c.isalnum() or c.isspace()])

df['processed_description'] = df['risk_description'].apply(preprocess_text)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Risk Feedback", layout="wide")
st.title("🤖 AI-Powered Risk Feedback for Mural")

with st.sidebar:
    st.header("🔧 Settings")
    num_clusters = st.slider("Number of Risk Themes (Clusters)", 5, 20, 10)
    severity_threshold = st.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)
    top_k = st.slider("Top Similar Risks to Retrieve", 1, 10, 5)
    st.markdown("---")
    if st.button("🔄 Pull Existing Sticky Notes from Mural"):
        headers = {'Authorization': f'Bearer {MURAL_API_TOKEN}'}
        mural_data = requests.get(
            f"https://api.mural.co/v1/workspaces/{MURAL_WORKSPACE_ID}/murals/{MURAL_BOARD_ID}/widgets",
            headers=headers
        )
        if mural_data.status_code == 200:
            widgets = mural_data.json().get("value", [])
            stickies = [w for w in widgets if w['type'] == 'sticky_note']
            content = [w['text'] for w in stickies]
            st.session_state['mural_notes'] = content
            st.success(f"Pulled {len(content)} sticky notes from Mural")
        else:
            st.error("Failed to pull stickies from Mural")

# Load mural stickies if already pulled
mural_notes = st.session_state.get("mural_notes", [])

# User risk input
st.subheader("1️⃣ Provide Risk Input")
user_input = st.text_area("Paste risks here (one per line) or pull from Mural sidebar.", height=200)

if not user_input and mural_notes:
    user_input = "\n".join(mural_notes)
    st.info("Using sticky notes pulled from Mural.")

# --- Embedding and Clustering ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
csv_embeddings = embedder.encode(df['processed_description'].tolist(), show_progress_bar=False)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(csv_embeddings)

# Build FAISS index
dimension = csv_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(csv_embeddings)

if st.button("🚀 Run Risk Feedback Analysis"):
    if user_input:
        human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
        human_embeddings = embedder.encode(human_risks)
        human_embeddings = np.array(human_embeddings)
        distances, indices = index.search(human_embeddings, top_k)
        similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]

        human_clusters = set([r['cluster'] for group in similar_risks for r in group])
        human_types = set([r['risk_type'] for group in similar_risks for r in group])
        human_stakeholders = set([r['stakeholder'] for group in similar_risks for r in group])

        missed_clusters = set(df['cluster']) - human_clusters
        missed_types = set(df['risk_type']) - human_types
        missed_stakeholders = set(df['stakeholder']) - human_stakeholders

        high_severity_df = df[df['severity'] >= severity_threshold]
        missed_high_severity = high_severity_df[~high_severity_df['cluster'].isin(human_clusters)]
        top_missed = missed_high_severity.sort_values(by='combined_score', ascending=False).head(5)

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

        # Store results
        session_id = str(uuid.uuid4())
        os.makedirs("sessions", exist_ok=True)
        with open(f"sessions/{session_id}.json", "w") as f:
            json.dump({
                "feedback": feedback,
                "top_missed_risks": top_missed.to_dict(orient="records")
            }, f)

        st.subheader("2️⃣ Feedback on Your Risk Analysis")
        st.markdown(feedback)
        st.markdown(f"[🔗 Reopen this session](?session={session_id})")

        st.session_state['missed_risks'] = top_missed.to_dict(orient="records")
        st.session_state['session_id'] = session_id
    else:
        st.warning("Please provide input risks or pull from Mural.")

# --- Post to Mural ---
if 'missed_risks' in st.session_state:
    missed_risks = st.session_state['missed_risks']
    session_id = st.session_state['session_id']

    st.subheader("3️⃣ Push Insights to Mural")
    if "posted_count" not in st.session_state:
        st.session_state.posted_count = 0

    st.markdown(f"**Posted {st.session_state.posted_count} of {len(missed_risks)} risks**")

    if st.button("📝 Post Next Missed Risk to Mural"):
        if st.session_state.posted_count < len(missed_risks):
            risk = missed_risks[st.session_state.posted_count]
            mural_url = f"https://your-streamlit-app.com/?session={session_id}"
            payload = {
                "x": 1000 + st.session_state.posted_count * 100,
                "y": 1000,
                "width": 300,
                "height": 150,
                "text": f"\ud83e\udde0 AI Risk Suggestion:\n{risk['risk_description']}\n\n[View Full Feedback]({mural_url})",
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
                st.success(f"Posted to Mural: {risk['risk_description']}")
                st.session_state.posted_count += 1
            else:
                st.error(f"Failed to post to Mural: {mural_response.text}")
        else:
            st.info("✅ All top risks have been posted.")
