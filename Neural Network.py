import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. สร้าง Dataset (ข้อมูลหอพักนักศึกษา)
data = {
    "ชื่อหอพัก": ["พรโชคชัย", "บ้านสวนสมบูรณ์", "บ้านมณีวรรณ", "บ้านชวนชม", "บ้านพักศิริบูรณ์", "นภาเพลส"],
    "ระยะทาง": [7.3, 7.5, 2.1, 2.4, 2.3, 6.0],
    "ห้องนอน": [1, 1, 1, 1, 1, 1],
    "แอร์": [0, 1, 1, 1, 1, 1],
    "เฟอร์นิเจอร์": [0, 1, 1, 1, 1, 1],
    "Wifi": [1, 1, 1, 0, 1, 1],
    "กล้องวงจรปิด": [1, 1, 1, 1, 1, 1],
    "คีย์การ์ด": [1, 1, 1, 1, 1, 1],
    "TV": [0, 1, 1, 1, 1, 1],
    "ที่จอดรถ": [1, 1, 1, 1, 2, 2],
    "ค่าเช่า": [2500, 3800, 5000, 4800, 7000, 6500]
}
df = pd.DataFrame(data)

# 2. แยก Features และ Target
X = df.drop(columns=["ชื่อหอพัก", "ค่าเช่า"])  # Features
y = df["ค่าเช่า"]  # Target

# 3. แบ่งข้อมูลเป็น Training และ Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. สร้างโมเดล (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)  # ฝึกโมเดล

# 5. ทดสอบโมเดล
y_pred = model.predict(X_test)

# 6. ประเมินผลโมเดล
mae = mean_absolute_error(y_test, y_pred)
print(f"ค่าความคลาดเคลื่อนสัมบูรณ์เฉลี่ย (MAE): {mae:.2f} บาท")

# 7. ลองทำนายราคาค่าเช่าหอพักใหม่
new_room_data = [[1.0, 1, 1, 1, 1, 1, 1, 1, 2]]  # ลดเหลือ 9 ค่าให้ตรงกับ X
new_room = pd.DataFrame(new_room_data, columns=X.columns)  # ใช้ X.columns เพื่อให้แน่ใจว่าคอลัมน์ตรงกัน
predicted_price = model.predict(new_room)
print(f"ราคาค่าเช่าที่คาดการณ์: {predicted_price[0]:,.2f} บาท")











