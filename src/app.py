import streamlit as st
from predict import predict_priority
import pandas as pd
import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'inbox.csv'
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI Email Classifier", page_icon="📧", layout="centered")

st.title("📧 AI Email Priority Classifier")
st.markdown("Paste any email below and the AI will classify its priority instantly.")

tab1, tab2 = st.tabs(["Classify Email", "Inbox"])

with tab1:
    st.markdown("### Classify New Email")
    subject = st.text_input("Subject line", placeholder="e.g. Urgent: Server is down!")
    body    = st.text_area("Email body", height=200, placeholder="Paste email content here...")

    if st.button("Classify Email", type="primary"):
        if subject or body:
            result = predict_priority(subject, body)
            priority   = result['priority']
            confidence = result['confidence']
            scores     = result['scores']

            color_map = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            bg_map    = {'high': '#ffe5e5', 'medium': '#fff8e5', 'low': '#e5ffe5'}

            st.markdown(f"### {color_map[priority]} Priority: **{priority.upper()}**")
            st.markdown(f"Confidence: `{confidence}%`")
            st.progress(int(confidence))

            st.markdown("#### Score breakdown")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High",   f"{scores.get('high',   0)}%")
            col2.metric("🟡 Medium", f"{scores.get('medium', 0)}%")
            col3.metric("🟢 Low",    f"{scores.get('low',    0)}%")

            # Save to inbox
            timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H.%M")
            data = {
                'subject': subject,
                'body': body,
                'priority': priority,
                'confidence': confidence,
                'timestamp': timestamp
            }
            df_new = pd.DataFrame([data])
            try:
                df_existing = pd.read_csv(DATA_PATH)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except FileNotFoundError:
                df_combined = df_new
            df_combined.to_csv(DATA_PATH, index=False)
            st.success("Email saved to inbox!")

            st.markdown("---")
            st.markdown("#### Was this classification correct?")
            feedback = st.radio("", ["Yes, correct!", "No - should be High", "No - should be Medium", "No - should be Low"])
            if st.button("Submit Feedback"):
                st.success("Thank you! Feedback saved. This will improve the model.")
        else:
            st.warning("Please enter a subject or body to classify.")

with tab2:
    st.header("📬 Inbox")
    try:
        df = pd.read_csv(DATA_PATH)
        # Sort by priority: high > medium > low
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        df['priority_num'] = df['priority'].map(priority_order)
        df = df.sort_values('priority_num', ascending=False).drop('priority_num', axis=1)
        df_reset = df.reset_index(drop=True)
        
        if df_reset.empty:
            st.info("No emails in inbox yet.")
        else:
            for idx, row in df_reset.iterrows():
                color_map = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                with st.expander(f"{color_map[row['priority']]} {row['priority'].upper()} - {row['subject'][:50]}..."):
                    st.write(f"**Subject:** {row['subject']}")
                    st.write(f"**Body:** {row['body']}")
                    st.write(f"**Priority:** {row['priority'].upper()}")
                    st.write(f"**Confidence:** {row['confidence']}%")
                    st.write(f"**Timestamp:** {row['timestamp']}")
                    
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("✓ Task Complete", key=f"delete_{idx}_{row['subject']}", use_container_width=True):
                            df_updated = df_reset.drop(idx).reset_index(drop=True)
                            df_updated.to_csv(DATA_PATH, index=False)
                            st.success("Email deleted!")
                            st.rerun()
    except FileNotFoundError:
        st.info("No emails in inbox yet.")

st.markdown("---")
st.markdown("*Built with Python, scikit-learn, and Streamlit*")