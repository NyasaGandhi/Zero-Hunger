import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


nltk.download('punkt')

# ====== Main Q&A Data (AI matched) ======
faq_data = {
    "What is zero hunger?": "Zero Hunger is SDG Goal 2, aiming to end hunger and promote sustainable agriculture.",
    "What is crop rotation?": """
- Crop rotation is the practice of growing different types of crops in the same field in a planned sequence over time
- It supports sustainable farming and helps achieve Zero Hunger by improving food production.""",
    "How to reduce food waste?": """Ways to Reduce Food Waste:
- Plan meals and buy only what you need  
- Store food properly  
- Use leftovers  
- Understand expiry dates  
- Serve smaller portions  
- Compost food scraps  
- Donate extra food  

    Reducing food waste helps will fight hunger and protect the planet.""",
    "What is organic farming?":"A method of farming that uses natural processes and organic inputs (like compost, manure, and biological pest control) to grow food without synthetic chemicals or genetically modified organisms (GMOs), aiming to produce healthy, sustainable, and nutritious food.",
    "Why is hunger still a problem?": """Hunger is still a problem because of: 
- Poverty
- Food waste
- Climate change
- Conflicts and wars
- Poor transport and storage
- Lack of education.""",
    "How can AI help in farming?": "AI can help with crop prediction, soil analysis, and pest detection.",
    "How can farmers improve yield?": "By using proper irrigation, fertilization, crop rotation, and tech-based solutions.",
    "What are sustainable farming practices?": "Using compost, reducing chemical use, and conserving water are sustainable methods."
}

# ====== Hardcoded Replies (Short common phrases) ======
custom_replies = {
    "hi": "Hello, How can I help you? 👋",
    "hello": "Hi there! How can I assist you today?",
    "hey": "Hey! Ask me anything about farming or hunger-related topics.",
    "thanks": "You're welcome! 😊",
    "thank you": "You're welcome! 😊",
    "ok": "Alright! Let me know if you have more questions.",
    "okay": "Okay! I'm here if you need anything else.",
    "bye": "bye! 👋 Stay aware and support Zero Hunger.",
    "goodbye": "Goodbye! 👋 Stay aware and support Zero Hunger."
}


questions = list(faq_data.keys())
answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# ====== Streamlit Setup ======
st.set_page_config(page_title="Zero Hunger Chatbot", page_icon="🌾", layout="centered")
tab1, tab2 = st.tabs(["🤖 Chatbot", "🌱 Crop Yield Predictor"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🌾 Zero Hunger Farming Chatbot")
st.caption("Ask about farming, food security, sustainable agriculture — or just say hi! 🙂")

# ====== Display chat history ======
for sender, msg in st.session_state.chat_history:
    with st.chat_message("user" if sender == "user" else "assistant"):
        st.markdown(msg)

# ====== Input like chatbot ======
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # ====== Check for custom replies first ======
    lower_input = user_input.lower().strip()
    if lower_input in custom_replies:
        bot_reply = custom_replies[lower_input]
    else:
        # ====== Fallback to AI matching (TF-IDF) ======
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)
        idx = np.argmax(similarity)

        if similarity[0][idx] > 0.2:
            bot_reply = answers[idx]
        else:
            bot_reply = "Sorry, I don’t have an answer to that yet."


    # Add delay before showing the bot's message
    time.sleep(1)  # ⏳ Bot "thinking" pause
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append(("bot", bot_reply))

# ====== Tab 2: Crop Yield Predictor ======
with tab2:
    st.title("🌱 Simple Crop Yield Predictor")
    st.caption("Estimate expected crop yield based on basic inputs.")

    crop = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize", "Sugarcane"])
    soil = st.selectbox("Select Soil Type", ["Loamy", "Sandy", "Clay"])
    area = st.number_input("Enter Land Area (in acres)", min_value=0.5, max_value=100.0, step=0.5)
    rainfall = st.number_input("Enter Rainfall (in mm)", min_value=0.0)
    fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0)

    if st.button("Predict Yield"):
        
        base_yield = {
            "Wheat": 2.5,
            "Rice": 3.0,
            "Maize": 2.8,
            "Sugarcane": 4.5
        }

        soil_factor = {"Loamy": 1.2, "Sandy": 0.9, "Clay": 1.0}
        rain_factor = rainfall / 500
        fert_factor = fertilizer / 100

        estimated_yield = base_yield[crop] * soil_factor[soil] * (0.5 + rain_factor + fert_factor)
        total_yield = estimated_yield * area

        st.success(f"🧑‍🌾 Estimated Yield for {crop} on {area} acres: **{total_yield:.2f} tons**")