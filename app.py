import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image

st.set_page_config(page_title="Facial Recognition App", layout="centered")
st.title("🧠 Facial Recognition + Verification")

tab1, tab2 = st.tabs(["🔍 Analyze Face", "👥 Verify Faces"])

# Tab 1: Analyze face
with tab1:
    uploaded_file = st.file_uploader("Upload a photo to analyze", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing face..."):
            try:
                result = DeepFace.analyze(img_path=np.array(image), actions=['age', 'gender', 'emotion'], enforce_detection=False)
                res = result[0]

                st.success("✅ Face Analyzed")
                st.write(f"**Age:** {res['age']}")
                st.write(f"**Gender:** {res['gender']}")
                st.write(f"**Dominant Emotion:** {res['dominant_emotion']}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# Tab 2: Verify faces
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        img1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        img2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="img2")

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)

        with st.spinner("Verifying..."):
            try:
                result = DeepFace.verify(img1_path=np.array(image1), img2_path=np.array(image2), enforce_detection=False)
                if result["verified"]:
                    st.success("✅ Faces Match")
                else:
                    st.error("❌ Faces Do NOT Match")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")