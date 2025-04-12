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
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlencode

# --- Configuration ---
st.set_page_config(page_title="AI Risk Feedback & Brainstorming", layout="wide")
st.title("🤖 AI-Powered Risk Analysis and Brainstorming for Mural")

# Load secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in .streamlit/secrets.toml with MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, MURAL_BOARD_ID, MURAL_REDIRECT_URI.")
    st.stop()

# Validate board ID
if MURAL_BOARD_ID != "1740767964926":
    st.warning("MURAL_BOARD_ID should be '1740767964926'. Update secrets.toml if incorrect.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- OAuth Functions ---
def get_authorization_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return f"https://app.mural.co/api/public/v1/authorization/oauth2/?{urlencode(params)}"

def exchange_code_for_token(code):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to exchange code for token: {response.status_code} - {response.text}")
        return None

def refresh_access_token(refresh_token):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to refresh token: {response.status_code} - {response.text}")
        return None

# --- Handle OAuth Flow ---
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None

# Check for authorization code in URL (callback)
query_params = st.query_params
auth_code = query_params.get("code")
if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data.get("refresh_token")
        st.session_state.token_expires_in = token_data.get("expires_in", 900)  # Default 15 min
        st.session_state.token_timestamp = pd.Timestamp.now().timestamp()
        # Clear query params to avoid reprocessing
        st.query_params.clear()
        st.success("Successfully authenticated with Mural!")

# Prompt user to authenticate if no token
if not st.session_state.access_token:
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize the app]({auth_url}) to access Mural.")
    st.stop()

# Refresh token if expired
if st.session_state.access_token:
    current_time = pd.Timestamp.now().timestamp()
    if (current_time - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        token_data = refresh_access_token(st.session_state.refresh_token)
        if token_data:
            st.session_state.access_token = token_data["access_token"]
            st.session_state.refresh_token = token_data.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = token_data.get("expires_in", 900)
            st.session_state.token_timestamp = pd.Timestamp.now().timestamp()

# --- Load Risk Dataset ---
csv_file = 'AI-Powered_Valuation_Enriched.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Dataset {csv_file} not found. Please upload the CSV file.")
    st.stop()

def preprocess_text(text):
    text = str(text).lower()
    return ''.join([c for c in text if c.isalnum() or c.isspace()])

df['processed_description'] = df['risk_description'].apply(preprocess_text)

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🔧 Settings")
    num_clusters = st.slider("Number of Clusters (Themes)", 5, 20, 10)
    severity_threshold = st.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)
    top_k = st.slider("Top Similar Risks to Retrieve", 1, 10, 5)
    st.markdown("---")
    st.subheader("📥 Pull Mural Notes")
    if st.button("🔄 Pull Sticky Notes from Mural"):
        try:
            headers = {'Authorization': f'Bearer {st.session_state.access_token}'}
            url = f"https://app.mural.co/api/public/v1/murals/{MURAL_BOARD_ID}/widgets"
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            mural_data = session.get(url, headers=headers)
            if mural_data.status_code == 200:
                widgets = mural_data.json().get("data", [])
                stickies = [w.get('text', '') for w in widgets if w.get('type') == 'sticky_note' and w.get('text')]
                st.session_state['mural_notes'] = stickies
                st.success(f"Pulled {len(stickies)} sticky notes from Mural.")
            else:
                st.error(f"Failed to pull from Mural: {mural_data.status_code} - {mural_data.text}")
                if mural_data.status_code == 401:
                    st.warning("OAuth token invalid or expired. Please re-authenticate.")
                    st.session_state.access_token = None
                    auth_url = get_authorization_url()
                    st.markdown(f"[Re-authorize the app]({auth_url}).")
                elif mural_data.status_code == 403:
                    st.warning("Access denied. Ensure your account is a collaborator on the mural: https://app.mural.co/t/aiimpacttesting2642/m/aiimpacttesting2642/1740767964926.")
                elif mural_data.status_code == 404:
                    st.warning("Mural not found. Confirm MURAL_BOARD_ID is '1740767964926'.")
                st.write("Raw API response:", mural_data.json())
        except Exception as e:
            st.error(f"Error connecting to Mural: {str(e)}")

# --- Initialize Model and Index ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
csv_embeddings = embedder.encode(df['processed_description'].tolist(), show_progress_bar=False)
df['cluster'] = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(csv_embeddings)
dimension = csv_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(csv_embeddings)

# --- Section 1: Input Risks ---
st.subheader("1️⃣ Input Risks")
default_notes = st.session_state.get('mural_notes', [])
default_text = "\n".join(default_notes) if default_notes else ""
user_input = st.text_area("Paste or edit your risks below:", value=default_text, height=200)

# --- Section 2: Generate Feedback ---
st.subheader("2️⃣ Generate Feedback on Risk Coverage")
if st.button("🔍 Generate Feedback"):
    if user_input.strip():
        human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
        human_embeddings = np.array(embedder.encode(human_risks))
        distances, indices = index.search(human_embeddings, top_k)
        similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]

        # Extract covered and missed themes
        covered_clusters = {r['cluster'] for group in similar_risks for r in group}
        covered_types = {r['risk_type'] for group in similar_risks for r in group}
        covered_stakeholders = {r['stakeholder'] for group in similar_risks for r in group}

        missed_clusters = set(df['cluster']) - covered_clusters
        missed_types = set(df['risk_type']) - covered_types
        missed_stakeholders = set(df['stakeholder']) - covered_stakeholders

        top_missed = df[(df['severity'] >= severity_threshold) & (~df['cluster'].isin(covered_clusters))]
        top_missed = top_missed.sort_values(by='combined_score', ascending=False).head(5)

        prompt = f"""
        You are an AI risk analysis expert. The user provided these risks: {', '.join(human_risks)}
        Identify what they missed based on risk database themes, risk types, stakeholders, and high-severity risks:

        Missed Clusters: {missed_clusters}
        Missed Risk Types: {missed_types}
        Missed Stakeholders: {missed_stakeholders}

        Top high-severity missed examples:
        {chr(10).join('- ' + r for r in top_missed['risk_description'].tolist())}
        Provide detailed, constructive feedback on where coverage is weak.
        """

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI risk advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content
            st.markdown("### 🧠 Feedback:")
            st.markdown(feedback)

            st.session_state['missed_risks'] = top_missed.to_dict(orient='records')
            st.session_state['feedback'] = feedback
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
    else:
        st.warning("Please enter or pull some risk input first.")

# --- Section 3: Post Missed Risks to Mural ---
if 'missed_risks' in st.session_state:
    st.subheader("3️⃣ Post Missed Risks to Mural")
    if "posted_count" not in st.session_state:
        st.session_state.posted_count = 0

    if st.button("📝 Post Next AI Suggestion to Mural"):
        missed_risks = st.session_state['missed_risks']
        idx = st.session_state.posted_count
        if idx < len(missed_risks):
            risk = missed_risks[idx]
            payload = {
                "x": 1000 + idx * 120,
                "y": 1000,
                "width": 300,
                "height": 150,
                "text": f"🧠 Missed Risk:\n{risk['risk_description']}\n(Severity: {risk['severity']})",
                "shape": "rectangle",
                "style": {"backgroundColor": "#FFFF99"},
                "type": "sticky_note"
            }
            headers = {
                'Authorization': f'Bearer {st.session_state.access_token}',
                'Content-Type': 'application/json'
            }
            url = f"https://app.mural.co/api/public/v1/murals/{MURAL_BOARD_ID}/widgets"
            try:
                session = requests.Session()
                retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                session.mount('https://', HTTPAdapter(max_retries=retries))
                res = session.post(url, headers=headers, json=payload)
                if res.status_code in [200, 201]:
                    st.success(f"Posted: {risk['risk_description']}")
                    st.session_state.posted_count += 1
                else:
                    st.error(f"Error posting to Mural: {res.status_code} - {res.text}")
                    if res.status_code == 401:
                        st.warning("OAuth token invalid. Please re-authenticate.")
                        st.session_state.access_token = None
                        auth_url = get_authorization_url()
                        st.markdown(f"[Re-authorize the app]({auth_url}).")
                    elif res.status_code == 403:
                        st.warning("Access denied for posting. Ensure collaborator status.")
            except Exception as e:
                st.error(f"Error posting to Mural: {str(e)}")
        else:
            st.info("✅ All missed risks posted.")

# --- Section 4: Brainstorming Assistant ---
st.subheader("4️⃣ Brainstorm with AI")
stakeholder_options = sorted(df['stakeholder'].dropna().unique())
risk_type_options = sorted(df['risk_type'].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    stakeholder = st.selectbox("Target Stakeholder (optional):", ["Any"] + stakeholder_options)
with col2:
    risk_type = st.selectbox("Target Risk Type (optional):", ["Any"] + risk_type_options)

if st.button("💡 Suggest AI-Generated Risks"):
    filt = df.copy()
    if stakeholder != "Any":
        filt = filt[filt['stakeholder'] == stakeholder]
    if risk_type != "Any":
        filt = filt[filt['risk_type'] == risk_type]
    top_suggestions = filt.sort_values(by='combined_score', ascending=False).head(5)

    suggestions = "\n".join(f"- {r}" for r in top_suggestions['risk_description'].tolist())

    prompt = f"""
    Generate brainstorming risk suggestions for an AI deployment based on these:
    {suggestions}

    Phrase them as creative suggestions to help a human identify overlooked risks.
    """

    try:
        result = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI brainstorming assistant for strategic risk workshops."},
                {"role": "user", "content": prompt}
            ]
        )
        brainstorm_output = result.choices[0].message.content
        st.markdown("### 🧠 AI Brainstorm Suggestions:")
        st.markdown(brainstorm_output)
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
