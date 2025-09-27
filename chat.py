import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import requests

# -----------------------------
# AI API Setup
# -----------------------------
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer YOUR_HF_API_KEY"}

def call_llm(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "error" in result:
            return f"Error: {result['error']}"
        else:
            return str(result)
    except Exception as e:
        return f"LLM call failed: {e}"

# -----------------------------
# Dataset
# -----------------------------
data = pd.DataFrame({
    "Block": ["A", "B", "C", "D"],
    "Recharge (MCM)": [120, 95, 60, 30],
    "Extraction (MCM)": [80, 70, 55, 40],
    "Stage (%)": [66.7, 73.7, 91.7, 133.3],
    "Category": ["Safe", "Safe", "Critical", "Over-Exploited"]
})

# -----------------------------
# Multilingual Support
# -----------------------------
language = st.sidebar.selectbox("🌐 Choose Language", ["English", "Hindi"])
translations = {
    "English": {
        "title": "💧 INGRES AI Driven ChatBOT",
        "tagline": "AI-powered groundwater insights for sustainable water management",
        "query": "💬 Ask about groundwater status:",
        "response": "Response",
        "tabs": ["Chat", "Data Report", "Visualization", "Map View"]
    },
    "Hindi": {
        "title": "💧 INGRES भूजल चैटबॉट",
        "tagline": "सतत जल प्रबंधन के लिए एआई-संचालित भूजल जानकारी",
        "query": "💬 भूजल स्थिति के बारे में पूछें:",
        "response": "उत्तर",
        "tabs": ["चैट", "डेटा रिपोर्ट", "दृश्य", "मानचित्र दृश्य"]
    }
}
t = translations[language]

# -----------------------------
# Page Styling
# -----------------------------
st.set_page_config(page_title="INGRES AI Driven ChatBOT", page_icon="💧", layout="wide")

st.markdown(
    f"""
    <div style="background: linear-gradient(90deg, #0077b6, #00b4d8); 
                padding: 20px; border-radius: 12px; text-align:center; color:white;">
        <h1>{t['title']}</h1>
        <p style="font-size:18px;">{t['tagline']}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Chat + Tabs
# -----------------------------
tabs = st.tabs(t["tabs"])

# --- Chat Tab ---
with tabs[0]:
    st.subheader(t["query"])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Rule-based + LLM
        def generate_response(query):
            if "safe" in query.lower():
                return data[data["Category"] == "Safe"]
            elif "critical" in query.lower():
                return data[data["Category"] == "Critical"]
            elif "over" in query.lower():
                return data[data["Category"] == "Over-Exploited"]
            elif "recharge" in query.lower():
                return data[["Block", "Recharge (MCM)"]]
            elif "extraction" in query.lower():
                return data[["Block", "Extraction (MCM)"]]
            else:
                return call_llm(query)

        result = generate_response(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": result})

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], pd.DataFrame):
                st.dataframe(msg["content"])
            else:
                st.write(msg["content"])

# --- Data Report Tab ---
with tabs[1]:
    st.subheader("📊 Groundwater Data Report")
    st.dataframe(data, use_container_width=True)

    # Quick Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Safe Blocks", (data["Category"] == "Safe").sum())
    col2.metric("⚠️ Critical Blocks", (data["Category"] == "Critical").sum())
    col3.metric("❌ Over-Exploited", (data["Category"] == "Over-Exploited").sum())

# --- Visualization Tab ---
with tabs[2]:
    st.subheader("📈 Groundwater Extraction Stage")
    fig, ax = plt.subplots()
    colors = ["green" if c == "Safe" else "orange" if c == "Critical" else "red" for c in data["Category"]]
    ax.bar(data["Block"], data["Stage (%)"], color=colors)
    ax.set_ylabel("Stage of Extraction (%)")
    ax.set_title("Groundwater Status by Block")
    st.pyplot(fig)

# --- Map Tab ---
with tabs[3]:
    st.subheader("🗺️ Groundwater Map View")
    df_map = pd.DataFrame({
        'lat': [28.6 + random.random() for _ in range(len(data))],
        'lon': [77.2 + random.random() for _ in range(len(data))],
        'Category': data["Category"]
    })
    st.map(df_map)
