import streamlit as st

st.set_page_config(
    page_title="Admission",
    page_icon="📊",
    layout="wide"
)

st.title("ข้อมูลการรับเข้าศึกษาต่อในมหาวิทยาลัย📊")
st.markdown("---")

st.markdown("""
สวัสดีครับ! 👋 แอปพลิเคชันนี้ถูกออกแบบมาเพื่อให้คุณได้สำรวจ *โมเดลการเรียนรู้ของเครื่อง* ที่แตกต่างกันสำหรับการทำนายการรับเข้าศึกษาต่อในมหาวิทลัย

""")

st.subheader("➡️ เลือกโมเดลด้านล่างเพื่อเริ่มการวิเคราะห์:")

st.page_link("pages/DTree.py", label="โมเดล Decision Tree 🌲", icon="1️⃣")
st.page_link("pages/KnnwithHeart.py", label="โมเดล K-Nearest Neighbors 🏘️", icon="2️⃣")
st.page_link("pages/NaiveBaye.py", label="โมเดล Naive Bayes 🧠", icon="3️⃣")

st.markdown("---")