import streamlit as st

st.set_page_config(
    page_title="AI เพื่อสุขภาพหัวใจ",
    page_icon="❤️",
    layout="wide"
)

st.title("ผู้ช่วย AI เพื่อสุขภาพหัวใจของคุณ ❤️📊")
st.markdown("---")

st.markdown("""
สวัสดีครับ! 👋 แอปพลิเคชันนี้ถูกออกแบบมาเพื่อให้คุณได้สำรวจ *โมเดลการเรียนรู้ของเครื่อง* ที่แตกต่างกันสำหรับการทำนายความเสี่ยงโรคหัวใจวาย

ไม่ว่าคุณจะเป็นผู้ที่ชื่นชอบด้านวิทยาศาสตร์ข้อมูลหรือแค่อยากรู้อยากเห็น ก็สามารถเลือกโมเดลการทำนายจากเมนูนำทางด้านซ้ายเพื่อเริ่มต้นได้เลย
""")

st.subheader("➡️ เลือกโมเดลด้านล่างเพื่อเริ่มการวิเคราะห์:")

st.page_link("pages/DTree.py", label="โมเดล Decision Tree 🌲", icon="1️⃣")
st.page_link("pages/KnnwithHeart.py", label="โมเดล K-Nearest Neighbors 🏘️", icon="2️⃣")
st.page_link("pages/NaiveBaye.py", label="โมเดล Naive Bayes 🧠", icon="3️⃣")

st.markdown("---")