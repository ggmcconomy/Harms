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
from datetime import datetime
import matplotlib.pyplot as plt

# Temporarily disable torch.classes to avoid Streamlit watcher error
sys.modules['torch.classes'] = None
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# --- Configuration ---
st.set_page_config(page_title="AI Risk Feedback & Brainstorming", layout="wide")
st.title("🤖 AI Risk Analysis Dashboard")

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

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Utility Functions ---
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    """Strip workspace prefix from mural ID if present."""
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
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
        st.error(f"Error cleaning HTML: {str(e)}")
        return ""

def log_feedback(risk_description, user_feedback, disagreement_reason=""):
    """Log user feedback to CSV."""
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "user_feedback": user_feedback,
        "disagreement_reason": disagreement_reason
    }
    feedback_df = pd.DataFrame([feedback_data])
    feedback_file = "feedback_log.csv"
    try:
        if os.path.exists(feedback_file):
            existing_df = pd.read_csv(feedback_file)
            feedback_df = pd.concat([existing_df, feedback_df], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)
    except Exception as e:
        st.error(f"Error logging feedback: {str(e)}")

def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Create a single bar chart for coverage."""
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(categories))
        plt.bar(x - 0.2, covered_counts, 0.4, label='Covered', color='#2ecc71')
        plt.bar(x + 0.2, missed_counts, 0.4, label='Missed', color='#e74c3c')
        plt.xlabel(title.split(' ')[-1])
        plt.ylabel('Number of Risks')
        plt.title(title)
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error creating chart {filename}: {str(e)}")
        return False

def create_coverage_charts(covered_stakeholders, missed_stakeholders, covered_types, missed_types, covered_clusters, missed_clusters):
    """Create bar charts for coverage visualization."""
    try:
        plt.style.use('ggplot')
    except Exception as e:
        st.warning(f"ggplot style failed: {str(e)}. Using default style.")
        plt.style.use('default')

    # Stakeholder Chart
    stakeholders = sorted(set(covered_stakeholders + missed_stakeholders))
    covered_counts = [covered_stakeholders.count(s) for s in stakeholders]
    missed_counts = [missed_stakeholders.count(s) for s in stakeholders]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    stakeholders = [stakeholders[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if stakeholders:
        create_coverage_chart("Stakeholder Coverage Gaps", stakeholders, covered_counts, missed_counts, 'stakeholder_coverage.png')
    else:
        st.warning("No stakeholder data to display.")

    # Risk Type Chart
    risk_types = sorted(set(covered_types + missed_types))
    covered_counts = [covered_types.count(t) for t in risk_types]
    missed_counts = [missed_types.count(t) for t in risk_types]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    risk_types = [risk_types[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if risk_types:
        create_coverage_chart("Risk Type Coverage Gaps", risk_types, covered_counts, missed_counts, 'risk_type_coverage.png')
    else:
        st.warning("No risk type data to display.")

    # Cluster Chart
    clusters = sorted(set(covered_clusters + missed_clusters))
    covered_counts = [covered_clusters.count(c) for c in clusters]
    missed_counts = [missed_clusters.count(c) for c in clusters]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    clusters = [clusters[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if clusters:
        create_coverage_chart("Cluster Coverage Gaps", [f"Cluster {c}" for c in clusters], covered_counts, missed_counts, 'cluster_coverage.png')
    else:
        st.warning("No cluster data to display.")

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
    with st.spinner("Authenticating with Mural..."):
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
                return response.json()
            else:
                st.error(f"Authentication failed: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None

def refresh_access_token(refresh_token):
    with st.spinner("Refreshing Mural token..."):
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
                return response.json()
            else:
                st.error(f"Token refresh failed: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Token refresh error: {str(e)}")
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
        if response.status_code == 200:
            return response.json().get("value", [])
        else:
            st.error(f"Failed to list murals: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error listing murals: {str(e)}")
        return []

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
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error verifying mural: {str(e)}")
        return False

# --- Handle OAuth Flow ---
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
        st.query_params.clear()
        st.success("Authenticated with Mural!")
        st.rerun()

if not st.session_state.access_token:
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize the app]({auth_url}) to access Mural.")
    st.info("Click the link above, log into Mural, and authorize.")
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

# --- Load Pre-Clustered Data ---
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
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🔧 Settings")
    num_clusters = st.slider("Number of Clusters (Themes)", 5, 20, 10)
    severity_threshold = st.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)
    st.markdown("---")
    st.subheader("📥 Mural Actions")
    custom_mural_id = st.text_input("Custom Mural ID (optional)", value=MURAL_BOARD_ID)
    if st.button("🔍 List Murals"):
        with st.spinner("Listing murals..."):
            murals = list_murals(st.session_state.access_token)
            if murals:
                st.write("Available Murals:", [{"id": m["id"], "title": m.get("title", "Untitled")} for m in murals])
            else:
                st.warning("No murals found.")
    if st.button("🔄 Pull Sticky Notes"):
        with st.spinner("Pulling sticky notes..."):
            try:
                headers = {'Authorization': f'Bearer {st.session_state.access_token}'}
                mural_id = custom_mural_id or MURAL_BOARD_ID
                if not verify_mural(st.session_state.access_token, mural_id):
                    mural_id = normalize_mural_id(mural_id)
                    if not verify_mural(st.session_state.access_token, mural_id):
                        st.error(f"Mural {mural_id} not found.")
                        st.stop()
                url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
                session = requests.Session()
                retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                session.mount('https://', HTTPAdapter(max_retries=retries))
                mural_data = session.get(url, headers=headers, timeout=10)
                if mural_data.status_code == 200:
                    widgets = mural_data.json().get("value", mural_data.json().get("data", []))
                    sticky_widgets = [w for w in widgets if w.get('type', '').replace(' ', '_').lower() == 'sticky_note']
                    stickies = []
                    for w in sticky_widgets:
                        raw_text = w.get('htmlText') or w.get('text') or ''
                        if raw_text:
                            cleaned_text = clean_html_text(raw_text)
                            if cleaned_text:
                                stickies.append(cleaned_text)
                    st.session_state['mural_notes'] = stickies
                    st.success(f"Pulled {len(stickies)} sticky notes.")
                else:
                    st.error(f"Failed to pull sticky notes: {mural_data.status_code}")
                    if mural_data.status_code == 401:
                        st.warning("OAuth token invalid. Please re-authenticate.")
                        st.session_state.access_token = None
                        auth_url = get_authorization_url()
                        st.markdown(f"[Re-authorize the app]({auth_url}).")
                    elif mural_data.status_code == 403:
                        st.warning("Access denied. Ensure collaborator access.")
                    elif mural_data.status_code == 404:
                        st.warning(f"Mural ID {mural_id} not found.")
            except Exception as e:
                st.error(f"Error connecting to Mural: {str(e)}")
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        st.rerun()

# --- Section 1: Input Risks ---
st.subheader("1️⃣ Input Risks")
st.write("Enter risks from Mural or edit below to analyze coverage.")
default_notes = st.session_state.get('mural_notes', [])
default_text = "\n".join(default_notes) if default_notes else ""
user_input = st.text_area("", value=default_text, height=200, placeholder="Enter risks, one per line.")

# --- Section 2: Generate Feedback ---
st.subheader("2️⃣ Generate Feedback")
st.write("Analyze risk coverage and identify gaps.")
num_missed_risks = st.slider("Number of Missed Risks to Show", 1, 5, 5)
if st.button("🔍 Generate Feedback"):
    with st.spinner("Analyzing risks..."):
        if user_input.strip():
            human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
            human_embeddings = np.array(embedder.encode(human_risks))
            distances, indices = index.search(human_embeddings, 5)
            similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]

            covered_clusters = {r['cluster'] for group in similar_risks for r in group}
            covered_types = {r['risk_type'] for group in similar_risks for r in group}
            covered_stakeholders = {r['stakeholder'] for group in similar_risks for r in group}

            missed_clusters = sorted(list(set(df['cluster']) - covered_clusters))
            missed_types = sorted(list(set(df['risk_type']) - covered_types))
            missed_stakeholders = sorted(list(set(df['stakeholder']) - covered_stakeholders))

            top_missed = df[(df['severity'] >= severity_threshold) & (~df['cluster'].isin(covered_clusters))]
            top_missed = top_missed.sort_values(by='combined_score', ascending=False).head(num_missed_risks)

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
                
                # Prepare data for coverage charts
                covered_stakeholder_list = [r['stakeholder'] for group in similar_risks for r in group]
                covered_type_list = [r['risk_type'] for group in similar_risks for r in group]
                covered_cluster_list = [r['cluster'] for group in similar_risks for r in group]
                missed_stakeholder_list = top_missed['stakeholder'].tolist()
                missed_type_list = top_missed['risk_type'].tolist()
                missed_cluster_list = top_missed['cluster'].tolist()

                create_coverage_charts(
                    covered_stakeholder_list, missed_stakeholder_list,
                    covered_type_list, missed_type_list,
                    covered_cluster_list, missed_cluster_list
                )

                st.session_state['missed_risks'] = top_missed.to_dict(orient='records')
                st.session_state['feedback'] = feedback
                st.session_state['coverage_data'] = {
                    'covered_stakeholders': covered_stakeholder_list,
                    'missed_stakeholders': missed_stakeholder_list,
                    'covered_types': covered_type_list,
                    'missed_types': missed_type_list,
                    'covered_clusters': bitcastaditya_cluster_list,
                    'missed_clusters': missed_cluster_list
                }

            except Exception as e:
                st.error(f"OpenAI API error: {str(e)}")
        else:
            st.warning("Please enter or pull some risks first.")

# --- Section 3: Coverage Visualization ---
if 'coverage_data' in st.session_state:
    st.subheader("3️⃣ Coverage Visualization")
    st.write("View gaps in risk coverage to identify weaknesses.")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_column_width=True)
        with col2:
            st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_column_width=True)
        with col3:
            st.image("cluster_coverage.png", caption="Cluster Gaps", use_column_width=True)
    except FileNotFoundError:
        st.error("Coverage charts failed to generate. Please try generating feedback again.")

# --- Section 4: Feedback and Suggested Risks ---
if 'feedback' in st.session_state:
    st.subheader("4️⃣ Feedback and Suggested Risks")
    st.write("Review AI feedback and suggested risks to improve coverage.")
    st.markdown("### Feedback:")
    st.markdown(st.session_state['feedback'])
    
    st.markdown("### Suggested Risks:")
    st.write("Vote on AI-suggested risks. Agree to add manually to Mural, or disagree with a reason.")
    
    for idx, risk in enumerate(st.session_state['missed_risks']):
        risk_key = f"risk_{idx}"
        short_text = risk['risk_description'][:200] + ("..." if len(risk['risk_description']) > 200 else "")
        
        st.markdown(f"**Risk {idx + 1}:** {short_text}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Agree", key=f"agree_{risk_key}"):
                log_feedback(risk['risk_description'], "agree")
                st.success("Thanks! Copy this risk to add it to Mural manually.")
        with col2:
            if st.button("👎 Disagree", key=f"disagree_{risk_key}"):
                st.session_state[f"show_disagree_{risk_key}"] = True
        
        if st.session_state.get(f"show_disagree_{risk_key}", False):
            with st.form(key=f"disagree_form_{risk_key}"):
                disagreement_reason = st.text_area("Why do you disagree?", key=f"reason_{risk_key}", height=100)
                if st.form_submit_button("Submit"):
                    if disagreement_reason.strip():
                        log_feedback(risk['risk_description'], "disagree", disagreement_reason)
                        st.success("Disagreement noted. Thanks for your input!")
                        st.session_state[f"show_disagree_{risk_key}"] = False
                    else:
                        st.error("Please provide a reason.")

# --- Section 5: Brainstorming Assistant ---
st.subheader("5️⃣ Brainstorm Risks")
st.write("Generate creative risk ideas and provide feedback.")
num_brainstorm_risks = st.slider("Number of Suggestions", 1, 5, 5)
stakeholder_options = sorted(df['stakeholder'].dropna().unique())
risk_type_options = sorted(df['risk_type'].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    stakeholder = st.selectbox("Target Stakeholder (optional):", ["Any"] + stakeholder_options)
with col2:
    risk_type = st.selectbox("Target Risk Type (optional):", ["Any"] + risk_type_options)

if st.button("💡 Generate Suggestions"):
    with st.spinner("Generating ideas..."):
        filt = df.copy()
        if stakeholder != "Any":
            filt = filt[filt['stakeholder'] == stakeholder]
        if risk_type != "Any":
            filt = filt[filt['risk_type'] == risk_type]
        top_suggestions = filt.sort_values(by='combined_score', ascending=False).head(num_brainstorm_risks)

        suggestions = "\n".join(f"- {r}" for r in top_suggestions['risk_description'].tolist())

        prompt = f"""
        Generate {num_brainstorm_risks} creative risk suggestions for an AI deployment based on these:
        {suggestions}

        Phrase them to help identify overlooked risks. Provide each suggestion as a concise bullet point.
        """

        try:
            result = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI brainstorming assistant for risk workshops."},
                    {"role": "user", "content": prompt}
                ]
            )
            brainstorm_output = result.choices[0].message.content
            # Parse bullet points into a list
            brainstorm_suggestions = [s.strip() for s in brainstorm_output.split('\n') if s.strip().startswith('- ')]
            brainstorm_suggestions = [s[2:].strip() for s in brainstorm_suggestions]
            st.session_state['brainstorm_suggestions'] = brainstorm_suggestions[:num_brainstorm_risks]
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")

# Display Brainstorming Suggestions with Feedback
if 'brainstorm_suggestions' in st.session_state:
    st.markdown("### Brainstorm Suggestions:")
    st.write("Vote on AI-generated ideas. Agree to add manually to Mural, or disagree with a reason.")
    
    for idx, suggestion in enumerate(st.session_state['brainstorm_suggestions']):
        suggestion_key = f"brainstorm_{idx}"
        short_text = suggestion[:200] + ("..." if len(suggestion) > 200 else "")
        
        st.markdown(f"**Idea {idx + 1}:** {short_text}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Agree", key=f"agree_{suggestion_key}"):
                log_feedback(suggestion, "agree")
                st.success("Thanks! Copy this idea to add it to Mural manually.")
        with col2:
            if st.button("👎 Disagree", key=f"disagree_{suggestion_key}"):
                st.session_state[f"show_disagree_{suggestion_key}"] = True
        
        if st.session_state.get(f"show_disagree_{suggestion_key}", False):
            with st.form(key=f"disagree_form_{suggestion_key}"):
                disagreement_reason = st.text_area("Why do you disagree?", key=f"reason_{suggestion_key}", height=100)
                if st.form_submit_button("Submit"):
                    if disagreement_reason.strip():
                        log_feedback(suggestion, "disagree", disagreement_reason)
                        st.success("Disagreement noted. Thanks for your input!")
                        st.session_state[f"show_disagree_{suggestion_key}"] = False
                    else:
                        st.error("Please provide a reason.")
