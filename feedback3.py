import os
import json
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
import sys
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode
from bs4 import BeautifulSoup

# Temporarily disable torch.classes to avoid Streamlit watcher error
sys.modules['torch.classes'] = None
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# --- Configuration ---
st.set_page_config(page_title="AI Risk Feedback & Brainstorming", layout="wide")
st.title("🤖 AI-Powered Risk Analysis and Brainstorming for Mural")

st.text("Starting app...")
st.write("Debug: All session state keys:", list(st.session_state.keys()))

# Load secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "aiimpacttesting2642")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in .streamlit/secrets.toml.")
    st.stop()
st.text("Secrets loaded.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
st.text("OpenAI client ready.")

# --- Utility Functions ---
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    """Strip workspace prefix from mural ID if present."""
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
    return mural_id

def denormalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    """Add workspace prefix to mural ID if not present."""
    prefix = f"{workspace_id}."
    if not mural_id.startswith(prefix):
        return f"{prefix}{mural_id}"
    return mural_id

def clean_html_text(html_text):
    """Strip HTML tags and clean text."""
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(separator=" ").strip()
        return text if text else ""
    except Exception as e:
        st.write("Debug: Error cleaning HTML:", str(e))
        return ""

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
    with st.spinner("Exchanging OAuth code for token..."):
        st.text("Contacting Mural OAuth server...")
        url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "redirect_uri": MURAL_REDIRECT_URI,
            "code": code,
            "grant_type": "authorization_code"
        }
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                st.text("Token received.")
                token_data = response.json()
                st.write("Debug: OAuth Scopes:", token_data.get('scope', 'Not set'))
                return token_data
            else:
                st.error(f"Failed to exchange code: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error exchanging code: {str(e)}")
            return None

def refresh_access_token(refresh_token):
    with st.spinner("Refreshing OAuth token..."):
        st.text("Refreshing token...")
        url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                st.text("Token refreshed.")
                token_data = response.json()
                st.write("Debug: OAuth Scopes (refreshed):", token_data.get('scope', 'Not set'))
                return token_data
            else:
                st.error(f"Failed to refresh token: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error refreshing token: {str(e)}")
            return None

# --- Mural API Functions ---
def list_murals(auth_token):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=headers, timeout=10)
        st.write("Debug: List Murals Status Code:", response.status_code)
        st.write("Debug: List Murals Raw Response:", response.text)
        if response.status_code == 200:
            murals = response.json().get("value", [])
            st.write("Debug: Parsed Murals:", murals)
            return murals
        else:
            st.error(f"Failed to list murals: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error listing murals: {str(e)}")
        return []

def create_mural(auth_token, workspace_id, room_id=1740767942646471, title="Test Risk Mural"):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    payload = {
        "workspaceId": workspace_id,
        "roomId": int(room_id),
        "title": title
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.post(url, headers=headers, json=payload, timeout=10)
        st.write("Debug: Create Mural Status Code:", response.status_code)
        st.write("Debug: Create Mural Response:", response.text)
        if response.status_code in [200, 201]:
            mural_id = response.json().get("value", {}).get("id")
            st.write("Debug: Created Mural ID:", mural_id)
            return mural_id
        else:
            st.error(f"Failed to create mural: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error creating mural: {str(e)}")
        return None

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=headers, timeout=10)
        st.write("Debug: Verify Mural Status Code:", response.status_code)
        st.write("Debug: Verify Mural Response:", response.text)
        return response.status_code == 200
    except Exception as e:
        st.write("Debug: Error verifying mural:", str(e))
        return False

# --- Handle OAuth Flow ---
st.text("Checking OAuth status...")
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expires_in = None
    st.session_state.token_timestamp = None

query_params = st.query_params
auth_code = query_params.get("code")
if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data.get("refresh_token")
        st.session_state.token_expires_in = token_data.get("expires_in", 900)
        st.session_state.token_timestamp = pd.Timestamp.now().timestamp()
        st.write("Debug: Access Token:", st.session_state.access_token[:10] + "...")
        st.rerun()

if not st.session_state.access_token:
    st.text("Waiting for Mural authentication...")
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize the app]({auth_url}) to access Mural.")
    st.info("Click the link above, log into Mural, and authorize. You’ll be redirected back here.")
    st.stop()

if st.session_state.access_token:
    current_time = pd.Timestamp.now().timestamp()
    if (current_time - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        token_data = refresh_access_token(st.session_state.refresh_token)
        if token_data:
            st.session_state.access_token = token_data["access_token"]
            st.session_state.refresh_token = token_data.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = token_data.get("expires_in", 900)
            st.session_state.token_timestamp = pd.Timestamp.now().timestamp()
st.text("OAuth ready.")

# --- Load Pre-Clustered Data ---
st.text("Loading clustered dataset...")
csv_file = 'AI-Powered_Valuation_Clustered.csv'
embeddings_file = 'embeddings.npy'
index_file = 'faiss_index.faiss'

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Clustered CSV {csv_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

try:
    csv_embeddings = np.load(embeddings_file)
except FileNotFoundError:
    st.error(f"Embeddings file {embeddings_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

try:
    index = faiss.read_index(index_file)
except FileNotFoundError:
    st.error(f"Index file {index_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

# Initialize embedder
st.text("Loading SentenceTransformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
st.text("Dataset and models loaded.")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🔧 Settings")
    num_clusters = st.slider("Number of Clusters (Themes)", 5, 20, 10)
    severity_threshold = st.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)
    top_k = st.slider("Top Similar Risks to Retrieve", 1, 10, 5)
    st.markdown("---")
    st.subheader("📥 Mural Actions")
    custom_mural_id = st.text_input("Custom Mural ID (optional)", value=MURAL_BOARD_ID)
    if st.button("🔍 List Murals"):
        with st.spinner("Listing murals..."):
            murals = list_murals(st.session_state.access_token)
            if murals:
                st.write("Available Murals:", [{"id": m["id"], "title": m.get("title", "Untitled"), "permissions": m.get("visitorsSettings", {})} for m in murals])
            else:
                st.warning("No murals found or error occurred. Check debug output above.")
    if st.button("🆕 Create Test Mural"):
        with st.spinner("Creating test mural..."):
            mural_id = create_mural(st.session_state.access_token, MURAL_WORKSPACE_ID)
            if mural_id:
                st.success(f"Created mural with ID: {mural_id}")
                st.session_state['temp_mural_id'] = mural_id
    if st.button("🔄 Pull Sticky Notes from Mural"):
        with st.spinner("Pulling sticky notes from Mural..."):
            try:
                headers = {'Authorization': f'Bearer {st.session_state.access_token}'}
                mural_id = custom_mural_id or st.session_state.get('temp_mural_id', MURAL_BOARD_ID)
                st.write("Debug: Trying mural ID:", mural_id)
                if not verify_mural(st.session_state.access_token, mural_id):
                    st.warning(f"Mural {mural_id} not found. Trying normalized ID...")
                    mural_id = normalize_mural_id(mural_id)
                    st.write("Debug: Trying normalized mural ID:", mural_id)
                url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
                session = requests.Session()
                retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                session.mount('https://', HTTPAdapter(max_retries=retries))
                st.write("Debug: API Headers:", headers)
                mural_data = session.get(url, headers=headers, timeout=10)
                st.write("Debug: Mural ID Used:", mural_id)
                st.write("Debug: Status Code:", mural_data.status_code)
                st.write("Debug: Full API Response:", mural_data.json())
                if mural_data.status_code == 200:
                    widgets = mural_data.json().get("value", mural_data.json().get("data", []))
                    st.write("Debug: Total Widgets:", len(widgets))
                    widget_types = [w.get('type', 'unknown') for w in widgets]
                    st.write("Debug: Widget Types:", widget_types)
                    sticky_widgets = [w for w in widgets if w.get('type', '').replace(' ', '_').lower() == 'sticky_note']
                    st.write("Debug: Sticky Notes:", sticky_widgets)
                    stickies = []
                    for w in sticky_widgets:
                        raw_text = w.get('htmlText') or w.get('text') or ''
                        st.write("Debug: Raw Sticky Text:", raw_text)
                        if raw_text:
                            cleaned_text = clean_html_text(raw_text)
                            st.write("Debug: Cleaned Sticky Text:", cleaned_text)
                            if cleaned_text:
                                stickies.append(cleaned_text)
                    st.write("Debug: Extracted Stickies:", stickies)
                    st.session_state['mural_notes'] = stickies
                    st.write("Debug: mural_notes in session state:", st.session_state['mural_notes'])
                    st.success(f"Pulled {len(stickies)} sticky notes from Mural.")
                else:
                    st.error(f"Failed to pull from Mural: {mural_data.status_code} - {mural_data.text}")
                    if mural_data.status_code == 401:
                        st.warning("OAuth token invalid. Please re-authenticate.")
                        st.session_state.access_token = None
                        auth_url = get_authorization_url()
                        st.markdown(f"[Re-authorize the app]({auth_url}).")
                    elif mural_data.status_code == 403:
                        st.warning("Access denied. Ensure your account is a collaborator with write access.")
                    elif mural_data.status_code == 404:
                        st.warning(f"Mural ID {mural_id} not found. Try creating a new mural or check permissions.")
                    st.write("Raw API response:", mural_data.json())
            except Exception as e:
                st.error(f"Error connecting to Mural: {str(e)}")
    if st.button("🗑️ Clear Session State"):
        st.session_state.clear()
        st.rerun()

# --- Section 1: Input Risks ---
st.subheader("1️⃣ Input Risks")
default_notes = st.session_state.get('mural_notes', [])
default_text = "\n".join(default_notes) if default_notes else ""
st.write("Debug: default_notes:", default_notes)
st.write("Debug: default_text:", default_text)
user_input = st.text_area("Paste or edit your risks below:", value=default_text, height=200)

# --- Section 2: Generate Feedback ---
st.subheader("2️⃣ Generate Feedback on Risk Coverage")
if st.button("🔍 Generate Feedback"):
    with st.spinner("Generating feedback..."):
        if user_input.strip():
            human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
            human_embeddings = np.array(embedder.encode(human_risks))
            distances, indices = index.search(human_embeddings, top_k)
            similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]

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
        with st.spinner("Posting to Mural..."):
            missed_risks = st.session_state['missed_risks']
            idx = st.session_state.posted_count
            if idx < len(missed_risks):
                risk = missed_risks[idx]
                # Shorten text to avoid length issues
                short_text = risk['risk_description'][:100] + ("..." if len(risk['risk_description']) > 100 else "")
                payload = {
                    "x": 1000,
                    "y": 1000,
                    "text": f"Missed Risk: {short_text}",
                    "type": "sticky note"
                }
                headers = {
                    'Authorization': f'Bearer {st.session_state.access_token}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                # Prioritize temp_mural_id for posting
                mural_id = st.session_state.get('temp_mural_id', custom_mural_id or MURAL_BOARD_ID)
                # Verify mural before posting
                st.write("Debug: Verifying mural ID:", mural_id)
                if not verify_mural(st.session_state.access_token, mural_id):
                    st.warning(f"Mural {mural_id} not accessible.")
                st.write("Debug: Trying POST with mural ID:", mural_id)
                url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/stickynote"
                try:
                    session = requests.Session()
                    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                    session.mount('https://', HTTPAdapter(max_retries=retries))
                    st.write("Debug: POST Headers:", headers)
                    st.write("Debug: POST Payload:", payload)
                    res = session.post(url, headers=headers, json=payload)
                    st.write("Debug: Post response (/stickynote):", res.status_code, res.json())
                    if res.status_code in [200, 201]:
                        st.success(f"Posted: {risk['risk_description'][:50]}...")
                        st.session_state.posted_count += 1
                    else:
                        st.error(f"Error posting to Mural (/stickynote): {res.status_code} - {res.text}")
                        # Try /elements endpoint
                        st.write("Debug: Trying /elements endpoint...")
                        url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/elements"
                        res = session.post(url, headers=headers, json=payload)
                        st.write("Debug: Post response (/elements):", res.status_code, res.json())
                        if res.status_code in [200, 201]:
                            st.success(f"Posted: {risk['risk_description'][:50]}...")
                            st.session_state.posted_count += 1
                        else:
                            st.error(f"Error posting to Mural (/elements): {res.status_code} - {res.text}")
                            # Try numeric ID
                            numeric_id = normalize_mural_id(mural_id)
                            st.write("Debug: Trying POST with numeric mural ID:", numeric_id)
                            url = f"https://app.mural.co/api/public/v1/murals/{numeric_id}/stickynote"
                            res = session.post(url, headers=headers, json=payload)
                            st.write("Debug: Post response (numeric ID, /stickynote):", res.status_code, res.json())
                            if res.status_code in [200, 201]:
                                st.success(f"Posted: {risk['risk_description'][:50]}...")
                                st.session_state.posted_count += 1
                            else:
                                st.error(f"Error posting to Mural (numeric ID, /stickynote): {res.status_code} - {res.text}")
                                if res.status_code == 401:
                                    st.warning("OAuth token invalid. Please re-authenticate.")
                                    st.session_state.access_token = None
                                    auth_url = get_authorization_url()
                                    st.markdown(f"[Re-authorize the app]({auth_url}).")
                                elif res.status_code == 403:
                                    st.warning("Access denied for posting. Ensure collaborator status.")
                                elif res.status_code == 404:
                                    st.warning(f"Mural ID {numeric_id} not found. Try creating a new mural.")
                            st.write("Raw post response:", res.json())
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
    with st.spinner("Generating suggestions..."):
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
