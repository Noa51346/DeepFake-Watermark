import streamlit as st
import os

st.title("🛡️ DeepFake Watermark Detector")
st.write("מערכת לאימות מקוריות וידאו")

uploaded_file = st.file_uploader("העלי קובץ תמונה לבדיקה (כרגע עובדים על פריים בודד)", type=['png'])

if uploaded_file:
    # שמירה זמנית של הקובץ
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp_image.png", caption="התמונה שהועלתה")

    if st.button("בדוק חתימה סמויה"):
        # כאן נקרא לפונקציה מתוך הקובץ השני
        from watermark_logic import decode_image

        result = decode_image("temp_image.png")
        st.success(f"התוצאה: {result}")