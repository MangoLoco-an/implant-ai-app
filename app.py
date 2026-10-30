import cv2
import streamlit as gr
import torch
from ultralytics import YOLO

# Nạp trọng số AI
model = YOLO("best.pt")

def chan_doan_anh(img):
    if img is None:
        return None

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

    return results.plot(conf=False)

demo = gr.Interface(
    fn=chan_doan_anh,
    inputs=gr.Image(type="numpy", label="Kéo thả ảnh X-quang vào đây"),
    outputs=gr.Image(type="numpy", label="Kết quả AI chẩn đoán"),
    title="Hệ thống AI Chẩn đoán Tiêu xương Implant",
    description="Mô hình YOLOv11 tự động phân tích tổn thương dạng S-M và vùng răng khỏe mạnh.",
)

if __name__ == "__main__":
    demo.launch()
