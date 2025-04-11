import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from sklearn.cluster import KMeans
from collections import Counter
from dotenv import load_dotenv
import os

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CSV file
csv_file = 'AI-Powered_Valuation_Enriched.csv'  # Update this if your CSV filename differs
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Could not find '{csv_file}'. Please upload or place it in the correct directory.")
    st.stop()


# Preprocess risk descriptions
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text


df['processed_description'] = df['risk_description'].apply(preprocess_text)

# Streamlit app
st.title("AI Deployment Risk Analysis Feedback Tool")

# Sidebar customization controls
st.sidebar.header("Analysis Settings")
num_clusters = st.sidebar.slider("Number of Themes (Clusters)", min_value=5, max_value=20, value=10,
                                 help="Set the number of risk themes to identify.")
severity_threshold = st.sidebar.slider("Severity Threshold", min_value=0.0, max_value=5.0, value=4.0, step=0.5,
                                       help="Filter risks with severity above this value.")
top_k = st.sidebar.number_input("Top Similar Risks to Retrieve", min_value=1, max_value=10, value=5,
                                help="Number of similar risks to show per input.")

# Instructions
st.write("""
**Instructions**:  
Enter your AI deployment risks below (one per line). Use the settings on the left to customize the analysis:  
- **Number of Themes**: Adjust how many risk themes are grouped.  
- **Severity Threshold**: Focus on high-severity risks above this level.  
- **Top Similar Risks**: Set how many similar risks to retrieve.  
Click "Run Analysis" to generate feedback on gaps in your risk analysis, including missed themes, risk types, stakeholders, and high-severity risks.
""")

# Text area for user input
human_risks_input = st.text_area("Your AI Deployment Risks", height=200,
                                 placeholder="e.g., 'System crashes due to overload'\n'Inaccurate predictions'")

# Run button
if st.button("Run Analysis"):
    if human_risks_input:
        human_risks = [risk.strip() for risk in human_risks_input.split('\n') if risk.strip()]
        if human_risks:
            # Create embeddings
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            csv_embeddings = embedder.encode(df['processed_description'].tolist(), show_progress_bar=True)

            # Cluster embeddings to identify themes
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(csv_embeddings)

            # Summarize themes from CSV
            cluster_summary = df.groupby('cluster').agg(
                most_common_risk_type=('risk_type', lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else 'N/A'),
                most_common_stakeholder=(
                'stakeholder', lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else 'N/A'),
                example_risks=('risk_description', lambda x: x.head(3).tolist())
            ).reset_index()

            # Build FAISS index for retrieval
            dimension = csv_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(csv_embeddings)


            # Retrieve similar risks
            def retrieve_similar_risks(human_risks, top_k=top_k):
                human_embeddings = embedder.encode(human_risks)
                distances, indices = index.search(human_embeddings, top_k)
                similar_risks = [df.iloc[idx].to_dict('records') for idx in indices]
                return similar_risks


            similar_risks = retrieve_similar_risks(human_risks)

            # What human risks cover
            human_clusters = set([sr['cluster'] for sublist in similar_risks for sr in sublist])
            human_risk_types = set([sr['risk_type'] for sublist in similar_risks for sr in sublist])
            human_stakeholders = set([sr['stakeholder'] for sublist in similar_risks for sr in sublist])

            # What’s missing from human risks
            missed_clusters = set(df['cluster']) - human_clusters
            missed_risk_types = set(df['risk_type']) - human_risk_types
            missed_stakeholders = set(df['stakeholder']) - human_stakeholders

            # Filter high-severity missed risks
            high_severity_df = df[df['severity'] >= severity_threshold]
            missed_high_severity_risks = high_severity_df[~high_severity_df['cluster'].isin(human_clusters)]
            top_missed_risks = missed_high_severity_risks.sort_values(by='combined_score', ascending=False).head(3)

            # Examples for missed themes
            missed_theme_examples = []
            for cluster in missed_clusters:
                summary = cluster_summary[cluster_summary['cluster'] == cluster].iloc[0]
                theme = f"{summary['most_common_risk_type']} risks for {summary['most_common_stakeholder']}"
                examples = ", ".join(summary['example_risks'])
                missed_theme_examples.append(f"- Theme: {theme}\n  Examples: {examples}")

            # Examples for missed risk types
            missed_risk_type_examples = []
            for rt in missed_risk_types:
                examples = df[df['risk_type'] == rt]['risk_description'].head(2).tolist
                missed_risk_type_examples.append(f"- Risk Type: {rt}\n  Examples: {', '.join(examples)}")

            # Examples for missed stakeholders
            missed_stakeholder_examples = []
            for sh in missed_stakeholders:
                examples = df[df['stakeholder'] == sh]['risk_description'].head(2).tolist()
                missed_stakeholder_examples.append(f"- Stakeholder: {sh}\n  Examples: {', '.join(examples)}")

            # High-severity missed risks
            high_severity_examples = []
            for _, row in top_missed_risks.iterrows():
                high_severity_examples.append(
                    f"- '{row['risk_description']}' (Severity: {row['severity']}, Combined Score: {row['combined_score']})")

            # Construct prompt for GPT-4
            prompt = f"""
            You are an AI risk analysis expert. A user has provided these AI deployment risks:
            {', '.join(human_risks)}

            Based on the risk database, here’s what they seem to have missed:

            **Missed Themes:**
            {chr(10).join(missed_theme_examples) if missed_theme_examples else 'None'}

            **Missed Risk Types:**
            {chr(10).join(missed_risk_type_examples) if missed_risk_type_examples else 'None'}

            **Missed Stakeholders:**
            {chr(10).join(missed_stakeholder_examples) if missed_stakeholder_examples else 'None'}

            **High-Severity Missed Risks:**
            {chr(10).join(high_severity_examples) if high_severity_examples else 'None'}

            Provide direct feedback on what the user appears to have missed, focusing on high-risk areas. 
            Use the examples to explain why these gaps matter and suggest how they can improve their analysis.
            """

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content

            # Display feedback
            st.subheader("Feedback on Your Risk Analysis")
            st.markdown(feedback)
        else:
            st.warning("Please enter at least one risk.")
    else:
        st.info("Enter your risks above and click 'Run Analysis' to get feedback.")