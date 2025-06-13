# 🛡️ Personal Protective Equipment (PPE) Detection – Helmet Classification & Detection

A deep learning-based system for real-time **helmet detection** using a two-stage YOLOv11 pipeline and a custom CNN classifier. The solution ensures PPE compliance by identifying persons and checking if they are wearing helmets in both static images and videos.

---

## 🧑‍💻 Group Members

- **Muneeb ur Rehman Khan** – (2024-MSDS-115)  

---

## 🚀 Project Description

This project presents a robust and scalable system to detect helmet usage in workplaces using computer vision. It uses a **two-stage approach** for maximum accuracy:

1. **Stage 1 – Person Detection**
   - Detects human figures using YOLOv11.
   - Reduces background noise and irrelevant detections.

2. **Stage 2 – Helmet Detection**
   - Detects helmets **only within** the regions containing people.
   - If a helmet is found on a detected person → marked **"PPE Compliant"**.
   - If no helmet is found → marked **"Not PPE Compliant"**.

---

## 📦 Features

- ✅ Real-time helmet detection on video streams  
- ✅ Binary classification (helmet / no-helmet) using CNN  
- ✅ YOLOv11-based object detection for fast and accurate inference  
- ✅ Visualizations: confusion matrix, precision-recall curve, F1 scores  
- ✅ Modular codebase (classification, detection, training)  
- ✅ Easily extendable to include vests, goggles, etc.

---

## 🎥 Demo

**Input Video:**  
`video/construction1.mp4`

**Output with annotations:**  
`video/output_resized.mp4`

---

## 📚 Documentation

- 📄 [PPE Detection Report (PDF)](Documentation/Personal%20Protective%20Equipment%20Detection.pdf)  
- 📊 [Presentation Slides (PPTX)](Slides/Personal%20Protective%20Equipment%20Detection.pptx)

---

## 🛠️ How to Run

### 🔧 Setup

```bash
pip install torch torchvision opencv-python matplotlib

cd ./code/
python main.py
