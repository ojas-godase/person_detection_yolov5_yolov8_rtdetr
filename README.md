# People Detection & Counting — YOLOv8 vs YOLOv5 vs RT-DETR

This repository presents a complete workflow for **real-time person detection, tracking, and counting** using three modern object detection architectures — **YOLOv8**, **YOLOv5**, and **RT-DETR**.  
The project includes **data preparation**, **exploratory data analysis (EDA)**, **training and evaluation**, and a fully interactive **Streamlit deployment** for real-world testing.

---

## Project Overview

- **Objective**: Build a robust, real-time people detection system and benchmark different state-of-the-art architectures on the same dataset.
- **Models Explored**:
  - [YOLOv8](https://github.com/ultralytics/ultralytics)
  - [YOLOv5](https://github.com/ultralytics/yolov5)
  - [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- **Deployment**: Interactive [Streamlit](https://streamlit.io/) web app for:
  - Image detection
  - Video detection & live frame-by-frame counting
  - People counting using line crossing with **DeepSORT** tracking

## Why We Built This Project & Practical Applications

Accurately detecting and counting people in images and videos is a **core computer vision challenge with direct real-world impact**. Traditional manual counting is inefficient and error-prone, while commercial solutions are often expensive or closed-source.  
This project aims to build a **transparent, open, and reproducible pipeline** that allows anyone to train and deploy state-of-the-art person detection models tailored to their needs.
We trained all three models — **YOLOv5, YOLOv8, and RT-DETR** — on a **custom person-only COCO dataset** using **Kaggle Notebooks with NVIDIA Tesla T4 GPUs**, making the training reproducible and cost-efficient.
By comparing three leading architectures — **YOLOv5, YOLOv8, and RT-DETR** — on a controlled, **person-focused dataset**, we not only understand their trade-offs but also create a ready-to-use system for:

- **Crowd analytics & safety** — monitor occupancy in public venues, stadiums, airports, and events to ensure safe capacity.
- **Retail & smart spaces** — count foot traffic, analyze customer flow, and optimize store layouts or staffing.
- **Security & surveillance** — detect people in restricted zones, trigger alerts, and enhance camera monitoring systems.
- **Transportation & urban planning** — measure pedestrian movement across crosswalks, metro entrances, and other public areas.
- **Research & benchmarking** — provide a reproducible, flexible baseline for further experimentation in object detection.

Our approach shows how to go from **raw COCO data → filtered person-only dataset → training YOLO/transformer models → real-time application deployment**.

---

## Dataset & Preprocessing

We used the **COCO 2017 dataset** but **filtered it to only the `person` class** to make the system task-focused.

- Original: 5000 images, 36,781 annotations, 80 categories.
- Processed: Person-only images with regenerated YOLO format labels.

### Key Steps

- **Data download & filtering** (keeping only `person` class)
- **Conversion to YOLO label format**
- **Train/validation split (80/20)**
- **EDA on COCO (person distribution, annotation stats)**

**EDA Visuals (add your images here):**

- _EDA — Category Frequency (insert image here)_
- _EDA — Sample annotated images after filtering (insert image here)_

---

## 3. Model Training & Evaluation

We trained **three models** under controlled settings for fair comparison.

---

### YOLOv8

- Framework: `ultralytics`
- Config: `yolov8-person.yaml`
- Epochs: 100
- Image size: 640×640
- Optimizer: SGD

**Training Visualizations:** _(add corresponding figures here)_

- _Training Loss Curves — insert image here_
- _Validation Metrics (Precision, Recall, mAP) — insert image here_
- _Precision-Recall Curve — insert image here_
- _Confusion Matrix — insert image here_

**Final Metrics (YOLOv8):**

```
Precision:           0.7333
Recall:              0.6202
mAP@50:              0.6795
mAP@50-95:           0.4401
```

---

### YOLOv5

- Framework: `ultralytics/yolov5`
- Config: `person_v5.yaml`
- Epochs: 100
- Image size: 640×640

**Training Visualizations:** _(add your YOLOv5 images here)_

- _Training Loss Curves — insert image here_
- _Validation Metrics — insert image here_
- _Precision-Recall Curve — insert image here_
- _Confusion Matrix — insert image here_

**Final Metrics (YOLOv5):**

```
Precision:           0.801
Recall:              0.637
mAP@50:              0.715
mAP@50-95:           0.484
```

---

### RT-DETR

- Framework: `ultralytics` (Real-Time DEtection TRansformer)
- Model: RT-DETR-Lite
- Epochs: 25 (fast convergence for testing)

**Training Visualizations:** _(add your RT-DETR images here)_

- _Training Loss Curves — insert image here_
- _Validation Metrics — insert image here_
- _Precision-Recall Curve — insert image here_
- _Confusion Matrix — insert image here_

**Final Metrics (RT-DETR):**

```
Precision:           0.6621
Recall:              0.5220
mAP@50:              0.5858
mAP@50-95:           0.3494
```

---

## Model Performance Comparison & Analysis

We benchmarked three state-of-the-art detectors — **YOLOv5**, **YOLOv8**, and **RT-DETR** — trained on the same **person-only subset** of COCO.

### Final Validation Metrics

| Model   | Precision | Recall    | mAP@50    | mAP@50–95 |
| ------- | --------- | --------- | --------- | --------- |
| YOLOv8  | 0.7333    | 0.6202    | 0.6795    | 0.4401    |
| YOLOv5  | **0.801** | **0.637** | **0.715** | **0.484** |
| RT-DETR | 0.6621    | 0.5220    | 0.5858    | 0.3494    |

---

### Interpretation & Takeaways

#### YOLOv5 — Mature and Highly Optimized

- Achieved the **best overall performance** across all metrics.
- High **precision (0.801)** means fewer false positives.
- Likely benefits from years of community-driven optimization.

#### YOLOv8 — Competitive but More Lightweight

- Only slightly behind YOLOv5 (precision down ~7%, mAP50-95 down ~9%).
- Easier to train, cleaner API, faster to deploy.

#### RT-DETR — Promising but Needs Longer Training

- Transformer-based, better at long-range context.
- Lower performance due to shorter training (25 epochs).
- Scales well with more training data & tuning.

---

### ⚖️ Practical Impact

- **Highest detection accuracy & stability** → **YOLOv5**
- **Modern tooling, faster deployment** → **YOLOv8**
- **Future-proof transformer approach** → **RT-DETR** (after longer training)

---

## 4. Deployment — Streamlit App

We developed a fully interactive **Streamlit application** for testing these models.

### Features

- Upload an **image** and detect people.
- Upload a **video** for frame-wise counting.
- Upload a **video with a virtual line** for entry/exit counting (DeepSORT tracking).

**UI Screenshots (add your images here):**

- _App main interface — insert image here_
- _Image detection example — insert image here_
- _Video detection example — insert image here_
- _Line-cross counting example — insert image here_

**Run the app:**

```bash
streamlit run app.py
```

Switch between models by changing:

```python
MODEL_PATH = "yolov8_person_best.pt"  # or yolov5_person_best.pt / rtdetr_person_best.pt
```

---

## 5. Installation

```bash
# Install dependencies
pip install ultralytics
pip install streamlit opencv-python-headless deep-sort-realtime
```

---

## 6. Repository Structure

```
.
├── data/                  # COCO processed dataset (person-only)
├── training_notebooks/    # Kaggle notebooks for YOLOv5, YOLOv8, RT-DETR
├── weights/               # Trained model weights
├── app/                   # Streamlit application
├── utils/                 # Helper scripts for preprocessing & EDA
└── README.md
```

---

## 7. Highlights & Contributions

- Performed **EDA on COCO** to extract & analyze the `person` class distribution.
- Designed **clean training pipelines** for YOLOv5, YOLOv8, and RT-DETR.
- Conducted **systematic performance benchmarking** (Precision, Recall, mAP).
- Developed a **real-time people counting app** with image/video/line-crossing support.
- Produced **visualizations** for model metrics, confusion matrices, and predictions.

---
# person_detection
