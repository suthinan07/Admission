import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


# โหลดชุดข้อมูล Heart
Heart=pd.read_csv('./data/Heart3.csv')
X =Heart.drop(columns=['HeartDisease'])
y =Heart.HeartDisease

model = GaussianNB()
model.fit(X, y)  # ฝึกโมเดลล่วงหน้า

# ตั้งค่าหน้าเว็บ Streamlit
st.title("Naïve Bayes Classifier - โรคหัวใจ")
st.write("ป้อนคุณสมบัติเพื่อทำนายประเภท")

# รับค่าจากผู้ใช้ผ่าน slider
A1=st.number_input("ข้อมูลอายุ")
A2=st.number_input("ข้อมูลเพศ")
A3=st.number_input("ข้อมูล3")
A4=st.number_input("ข้อมูล4")
A5=st.number_input("ข้อมูล5")
A6=st.number_input("ข้อมูล6")
A7=st.number_input("ข้อมูล7")
A8=st.number_input("ข้อมูล8")
A9=st.number_input("ข้อมูล9")
A10=st.number_input("ข้อมูล10")
A11=st.number_input("ข้อมูล11")

# สร้างอาร์เรย์ข้อมูลจากค่าที่ป้อน
input_data = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])

# ทำนายผลลัพธ์
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# แสดงผลลัพธ์
st.subheader("ผลลัพธ์ที่ได้:")
st.write(f"ท่านเป็นโรคหัวใจหรือไม่: **{prediction[0]}**")

# แสดงความน่าจะเป็นของแต่ละคลาส
st.subheader("ความน่าจะเป็นของแต่ละประเภท:")
df_proba = pd.DataFrame(prediction_proba, columns=['0','1'])
st.dataframe(df_proba.style.format("{:.2%}"))
