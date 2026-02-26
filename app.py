import streamlit as st
import cv2
import os
import numpy as np
from datetime import date
import xlrd
from xlutils.copy import copy as xl_copy
from PIL import Image

st.title("📸 Face Recognition Attendance System")

# ======================
# TRAINING DATA
# ======================

labels = {"Student 1": 0, "Student 2": 1}

faces = []
ids = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for name, label in labels.items():
    img = cv2.imread(f"{name}.png")
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        faces.append(face_img)
        ids.append(label)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))

# ======================
# EXCEL SETUP
# ======================

excel_file = "attendance_excel.xls"

if not os.path.exists(excel_file):
    import xlwt
    wb = xlwt.Workbook()
    wb.add_sheet("Sheet1")
    wb.save(excel_file)

rb = xlrd.open_workbook(excel_file, formatting_info=True)
wb = xl_copy(rb)

lecture = st.text_input("Enter Lecture/Subject Name")

if "marked" not in st.session_state:
    st.session_state.marked = []
if "row" not in st.session_state:
    st.session_state.row = 1

if lecture:
    invalid = ['\\','/','*','?','[',']',':']
    for c in invalid:
        lecture = lecture.replace(c, "_")

    sheet = wb.add_sheet(lecture)
    sheet.write(0, 0, "Name")
    sheet.write(0, 1, str(date.today()))

    st.write("### 📷 Capture Image")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label, conf = recognizer.predict(face_img)

            name = "Unknown"
            for n, l in labels.items():
                if l == label:
                    name = n

            if name != "Unknown" and name not in st.session_state.marked:
                sheet.write(st.session_state.row, 0, name)
                sheet.write(st.session_state.row, 1, "Present")
                wb.save(excel_file)

                st.session_state.marked.append(name)
                st.session_state.row += 1

                st.success(f"Attendance Marked: {name}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
