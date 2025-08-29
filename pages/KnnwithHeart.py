from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายโอกาสการรับเข้าเรียนด้วย K-Nearest Neighbor')

# โหลดข้อมูล
dt = pd.read_csv("./data/data1.csv")

# สร้าง target ใหม่ (0/1)
dt['AdmitClass'] = (dt['Chance of Admit '] >= 0.75).astype(int)

# ตัดคอลัมน์ที่ไม่ใช้
X = dt.drop(['Serial No.', 'Chance of Admit ', 'AdmitClass'], axis=1)
y = dt['AdmitClass']

col1, col2 = st.columns(2)
if st.button("ทำนายผล", key="predict_btn"):
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[A1,A2,A3,A4,A5,A6,A7]])
    out = Knn_model.predict(x_input)

    st.write("ผลลัพธ์ที่ได้จากโมเดล:", out)  # debug

    if len(out) > 0 and out[0] == 1:
        st.success("🎉 มีโอกาสสูงที่จะได้รับการ Admit")
        st.markdown("![Admit](https://cdn-icons-png.flaticon.com/512/190/190411.png)")
    else:
        st.error("❌ โอกาสต่ำที่จะได้รับการ Admit")
        st.markdown("![Not Admit](https://cdn-icons-png.flaticon.com/512/753/753345.png)")



st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", X.columns)

st.write(f"### 🎯 Boxplot: {feature} แยกตามโอกาสการ Admit")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='AdmitClass', y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt.drop('Serial No.', axis=1), hue='AdmitClass')
    st.pyplot(fig2)

st.subheader("🔮 ทำนายข้อมูลใหม่")

A1 = st.number_input("กรุณาใส่ค่า GRE Score")
A2 = st.number_input("กรุณาใส่ค่า TOEFL Score")
A3 = st.number_input("กรุณาใส่ค่า University Rating")
A4 = st.number_input("กรุณาใส่ค่า SOP")
A5 = st.number_input("กรุณาใส่ค่า LOR")
A6 = st.number_input("กรุณาใส่ค่า CGPA")
A7 = st.number_input("กรุณาใส่ค่า Research (0=No, 1=Yes)")

if st.button("ทำนายผล"):
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[A1,A2,A3,A4,A5,A6,A7]])
    out = Knn_model.predict(x_input)

    if out[0] == 1:
        st.success("🎉 มีโอกาสสูงที่จะได้รับการ Admit")
   
