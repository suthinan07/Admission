import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="การทำนายโรคหัวใจวายDtree", page_icon="❤️")

# --- Header Section ---
st.title("🩺 การพยากรณ์โรคหัวใจวายด้วย Decision Tree")
st.markdown("---")
st.markdown("แอปพลิเคชันนี้ใช้โมเดล *Decision Tree* ในการวิเคราะห์ข้อมูลและทำนายความเสี่ยงของภาวะหัวใจวาย")

# --- Data Loading Section ---
st.subheader("📊 ข้อมูลที่ใช้ในการฝึกโมเดล")

try:
    # ✅ โหลดไฟล์จากโฟลเดอร์หลัก (เพราะโค้ดรันอยู่ใน pages/)
    df = pd.read_csv('./data/data1.csv')

    st.info("✅ โหลดข้อมูลจากไฟล์ *'adm_data.csv'* เรียบร้อยแล้ว")
    st.write("ตัวอย่าง 10 แถวแรกของชุดข้อมูล:")
    st.dataframe(df.head(10))

    # แสดงคอลัมน์ทั้งหมดให้ผู้ใช้ตรวจสอบ
    st.write("📌 คอลัมน์ที่พบในไฟล์ CSV:")
    st.write(list(df.columns))

    # --- Data Preparation ---
    target_column = 'Research'   # ❗ เปลี่ยนตรงนี้ถ้า target column ในไฟล์ชื่ออื่น เช่น 'Disease' หรือ 'Outcome'
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

    # --- Model Training ---
    with st.spinner('กำลังฝึกโมเดล Decision Tree...'):
        ModelDtree = DecisionTreeClassifier()
        dtree = ModelDtree.fit(x_train, y_train)
    st.success("✨ ฝึกโมเดลสำเร็จ!")

    # --- User Input Section ---
    st.subheader("✍️ ป้อนข้อมูลเพื่อพยากรณ์")
    st.markdown("---")

    # สร้าง input ตาม features จริง ๆ
    user_input = {}
    for feature in features:
        if df[feature].dtype in ['int64', 'float64']:
            user_input[feature] = st.number_input(f"ป้อนค่า {feature}", value=0)
        else:
            user_input[feature] = st.text_input(f"ป้อนค่า {feature}")

    if st.button("พยากรณ์ผล", type="primary"):
        x_input = pd.DataFrame([user_input])
        y_predict2 = dtree.predict(x_input)

        st.write("---")
        st.subheader("### 💡 ผลการพยากรณ์:")
        st.write(f"ค่าที่ได้จากโมเดล: **{y_predict2[0]}**")

    # --- Model Performance & Visualization ---
    st.markdown("---")
    st.subheader("📈 ประสิทธิภาพของโมเดล")
    y_predict = dtree.predict(x_test)
    score = accuracy_score(y_test, y_predict)

    st.metric(label="ความแม่นยำของโมเดล (Accuracy Score)", value=f"{int(score * 100)} %")

    st.subheader("🌳 แผนผัง Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 15))
    tree.plot_tree(dtree, feature_names=features, class_names=True, ax=ax,
                   filled=True, rounded=True, fontsize=10)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ *ไม่พบไฟล์ 'adm_data.csv'* กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกับโค้ดหรือใช้ path ที่ถูกต้อง")
except KeyError as e:
    st.error(f"❌ *เกิดข้อผิดพลาด: ไม่พบคอลัมน์ '{e}'* กรุณาตรวจสอบว่า target_column ตั้งถูกต้อง")
except Exception as e:
    st.error(f"❌ *เกิดข้อผิดพลาด*: {e}")
