# 📘 Task 1: Cross-Camera Player Mapping

## Project Overview

This project uses object detection (YOLOv8) and a simple spatial matching algorithm to map players consistently across two different camera angles of the same sports footage.

---

## Folder Structure

```
player_reid_task1/
├── main.py
├── utils/
│   └── matcher.py
├── models/
│   └── best.pt
├── data/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── output/
│   ├── broadcast_output.mp4
│   └── tacticam_output.mp4
├── requirements.txt
├── README.md
└── report.md
```

---

## Setup Instructions

### 1. Install Python

Python 3.8+ recommended.

### 2. Create and activate virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Place the following files in `data/` folder:

* `broadcast.mp4`
* `tacticam.mp4`

Download the custom YOLO model and place it in `models/best.pt`.

Then run:

```bash
python main.py
```

Annotated outputs will be saved to the `output/` folder.

---

## Download Required Files

- [Download best.pt](https://drive.google.com/file/d/1DbR7AF0AoREkiSCi2ctm8y_CZ5aRZt9a/view?usp=sharing)

---

## Notes

* Only one frame is processed from each video for basic mapping.
* Matching is based on center-point distance (can be upgraded with appearance matching or embeddings).
* Modify `matcher.py` for more advanced re-ID logic.
