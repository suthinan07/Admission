import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# โหลดชุดข้อมูล Admission
adm = pd.read_csv('./data/data1.csv')

# สร้าง target ใหม่ (0=ไม่ Admit, 1=Admit) จาก Chance of Admit
adm['AdmitClass'] = (adm['Chance of Admit '] >= 0.75).astype(int)

# กำหนด X, y
X = adm.drop(columns=['Serial No.', 'Chance of Admit ', 'AdmitClass'])
y = adm['AdmitClass']

# สร้างและฝึกโมเดล
model = GaussianNB()
model.fit(X, y)

# ตั้งค่าหน้าเว็บ Streamlit
st.title("Naïve Bayes Classifier - ทำนายการรับเข้าศึกษา")
st.write("ป้อนคุณสมบัติเพื่อทำนายโอกาสการ Admit")

# รับค่าจากผู้ใช้
A1 = st.number_input("GRE Score", min_value=0)
A2 = st.number_input("TOEFL Score", min_value=0)
A3 = st.number_input("University Rating", min_value=0)
A4 = st.number_input("SOP", min_value=0.0, step=0.5)
A5 = st.number_input("LOR", min_value=0.0, step=0.5)
A6 = st.number_input("CGPA", min_value=0.0, step=0.01)
A7 = st.number_input("Research (0=No, 1=Yes)", min_value=0, max_value=1)

# ปุ่มทำนาย
if st.button("ทำนายผล", key="predict_btn"):
    input_data = np.array([[A1, A2, A3, A4, A5, A6, A7]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # แสดงผลลัพธ์
    st.subheader("ผลลัพธ์ที่ได้:")
    if prediction[0] == 1:
        st.success("🎉 มีโอกาสสูงที่จะได้รับการ Admit")
    else:
        st.error("❌ โอกาสต่ำที่จะได้รับการ Admit")

    # แสดงความน่าจะเป็นของแต่ละประเภท
    st.subheader("ความน่าจะเป็นของแต่ละประเภท:")
    df_proba = pd.DataFrame(prediction_proba, columns=['Not Admit (0)','Admit (1)'])
    st.dataframe(df_proba.style.format("{:.2%}"))
