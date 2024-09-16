import cv2
import numpy as np
import streamlit as st
import imutils
import easyocr

# تهيئة قارئ easyocr
reader = easyocr.Reader(['en'])

# عنوان التطبيق في Streamlit
st.title("التعرف التلقائي على لوحة الأرقام")

# تحميل صورة باستخدام Streamlit
uploaded_file = st.file_uploader("اختر صورة سيارة...", type="jpg")

if uploaded_file is not None:
    # تحويل الملف المحمل إلى صورة باستخدام OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # معالجة الصورة
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB), caption='صورة باللونين الرمادي والأبيض', use_column_width=True)

    # فلترة ثنائية الجانب وكشف الحواف باستخدام Canny
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    st.image(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB), caption='صورة مع الحواف', use_column_width=True)

    # العثور على الكنتورات
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourss = imutils.grab_contours(keypoints)
    contours = sorted(contourss, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        # إنشاء قناع لموقع لوحة الأرقام
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], -1, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # عرض الصورة الجديدة مع الكنتور
        st.image(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), caption='تم تحديد لوحة الأرقام', use_column_width=True)

        # قص لوحة الأرقام من الصورة
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # عرض الصورة المقتطعة
        st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption='لوحة الأرقام المقتطعة', use_column_width=True)

        # استخدام easyocr لاستخراج النص من الصورة المقتطعة
        result = reader.readtext(cropped_image)

        # عرض النص المستخرج
        if result:
            plate_text = result[0][-2]  # استخراج النص المكتشف من نتيجة easyocr
            st.write(f"النص المستخرج من لوحة الأرقام: {plate_text}")

            # إضافة النص المستخرج إلى ملف نصي
            with open("ملف_النص_المستخرج_من_لوحة_السيارات.txt", "a") as file:
                file.write(f"Extracted Number Plate Text: {plate_text}\n")

            st.write("تم إضافة نص لوحة الأرقام إلى ملف_النص_المستخرج_من_لوحة_السيارات")

            # قراءة الملف وتحضيره للتنزيل
            with open("ملف_النص_المستخرج_من_لوحة_السيارات.txt", "r") as file:
                file_contents = file.read()

            # زر لتحميل ملف النصوص
            st.download_button(
                label="تحميل ملف النصوص",
                data=file_contents,
                file_name="لوحات_السيارات.txt",
                mime="text/plain"
            )
        else:
            st.write("لم يتم اكتشاف أي نص في لوحة الأرقام.")
    else:
        st.write("لم يتم اكتشاف لوحة الأرقام.")
