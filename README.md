# 🧠 Real-Time Object Detection with OpenCV + MobileNetSSD

Welcome to the future, where your webcam isn’t just watching — it's _thinking_.  
This project turns your plain ol' webcam into a real-time object detector that can spot anything from a **cat** to a **bottle**, or even your **chair** (yes, your chair 😄).

---

## 🚀 What It Does

Using OpenCV and a pre-trained **MobileNetSSD** model, this script detects objects from your webcam stream and draws bounding boxes with confidence scores — all in real-time.

No TensorFlow drama. No GPU drama. Just run it and watch your webcam show off.

---

## 🧠 Objects It Can Detect

The model recognizes 20 objects from the Pascal VOC dataset, including:

- `person`
- `car`, `bus`, `motorbike`
- `dog`, `cat`, `bird`, `cow`, `horse`, `sheep`
- `chair`, `sofa`, `diningtable`, `tvmonitor`
- `bottle`, `pottedplant`, `aeroplane`, `train`, `boat`, `bicycle`

You’ll be surprised how often you see random things it can detect just lying around your room.

---

## 🎯 Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Numpy (`pip install numpy`)
- Webcam (unless you’ve got magic powers)

---

## 🛠️ Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/Codegrammer999/simple-object-detection.git
   cd simple-object-detection
   ```
