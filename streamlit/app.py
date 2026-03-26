import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="TruthScope", layout="wide")

# Custom CSS (Premium Look)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00f5d4;
}
.subtitle {
    text-align: center;
    color: #a0aec0;
    margin-bottom: 20px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: #1c1f26;
    box-shadow: 0px 0px 20px rgba(0,255,200,0.15);
}
.fake {
    color: #ff4b5c;
    font-size: 28px;
    font-weight: bold;
}
.real {
    color: #00f5d4;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">🧠 TruthScope</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fake News Detection Dashboard</div>', unsafe_allow_html=True)

# Input Section
st.markdown("### ✍️ Enter News Content")
user_input = st.text_area("Paste your news here...", height=200)

# Analyze Button
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": user_input}
            )

            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            status = result["status"]
            fake_prob = result.get("fake_prob", 0.5)
            real_prob = result.get("real_prob", 0.5)

            # Layout
            col1, col2 = st.columns(2)

            # LEFT: Prediction Card
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if prediction == "Fake":
                    st.markdown(f'<div class="fake">🚨 {prediction}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="real">✅ {prediction}</div>', unsafe_allow_html=True)

                st.markdown(f"**Confidence:** {confidence}")
                st.markdown(f"**Status:** {status}")

                st.markdown('</div>', unsafe_allow_html=True)

            # RIGHT: Visual Analytics
            with col2:

                # Confidence Meter
                st.markdown("### 📊 Confidence Meter")
                st.progress(confidence)

                if confidence < 0.7:
                    st.warning("⚠️ Model is uncertain. Interpret carefully.")
                else:
                    st.success("✔️ High confidence prediction.")


                # Truth Spectrum (Fake vs Real)
                st.markdown("### ⚖️ Truth Spectrum")

                fig, ax = plt.subplots(figsize=(6, 1))

                ax.barh([""], fake_prob, color="#ff4b5c")
                ax.barh([""], real_prob, left=fake_prob, color="#00f5d4")

                ax.text(fake_prob/2, 0, f"Fake {fake_prob:.2f}", 
                        ha='center', va='center', color='white')

                ax.text(fake_prob + real_prob/2, 0, f"Real {real_prob:.2f}", 
                        ha='center', va='center', color='black')

                ax.set_xlim(0, 1)
                ax.axis('off')

                st.pyplot(fig)

        except:
            st.error("⚠️ API not running. Start FastAPI first.")