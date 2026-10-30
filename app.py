import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="AI Chẩn Đoán Tiêu Xương Implant", layout="centered")
st.title("Hệ thống AI Chẩn đoán Tiêu xương Implant")
st.write("Tải ảnh X-quang lên để mô hình YOLOv11 tự động khoanh vùng tổn thương.")

# Nạp mô hình vào bộ nhớ đệm để không bị tải lại mỗi lần bấm
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Chọn ảnh X-quang (JPG, PNG, BMP)...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Chuyển đổi file tải lên thành mảng numpy cho OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model(img, conf=0.15, verbose=False)[0]

    # Thuật toán lọc nhãn Healthy trùng đè
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy
        classes = results.boxes.cls
        benh_ly_idx = torch.where(classes > 0)[0]
        healthy_idx = torch.where(classes == 0)[0]
        keep_idx = benh_ly_idx.tolist()

        for h_idx in healthy_idx:
            hx1, hy1, hx2, hy2 = boxes[h_idx]
            h_area = (hx2 - hx1) * (hy2 - hy1)
            is_overlap = False
            for b_idx in benh_ly_idx:
                bx1, by1, bx2, by2 = boxes[b_idx]
                ix1, iy1 = max(hx1, bx1), max(hy1, by1)
                ix2, iy2 = min(hx2, bx2), min(hy2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_area / h_area) > 0.2:
                        is_overlap = True
                        break
            if not is_overlap:
                keep_idx.append(h_idx.item())
        results = results[keep_idx]

    res_img = results.plot(conf=False)
    # Chuyển hệ màu BGR của OpenCV sang RGB để hiển thị đúng màu trên Web
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    st.image(res_img_rgb, caption="Kết quả AI Chẩn Đoán", use_container_width=True)
