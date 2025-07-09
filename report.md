# Project Report: Task 1 - Cross-Camera Player Mapping

## Objective

To ensure players are consistently identified across two different camera angles of the same gameplay.

## Approach

1. **Object Detection**: Used YOLOv8 (fine-tuned model) to detect players in both `broadcast.mp4` and `tacticam.mp4`.
2. **Feature Extraction**: Extracted bounding box center coordinates as spatial features.
3. **Player Matching**: Used `scipy.spatial.distance.cdist` to compute distance between player centers in the two views.
4. **ID Assignment**: Matched the closest pairs to assign consistent player IDs across both videos.

## Techniques Tried

* Center-point distance matching (fast and simple)
* Class-based filtering (only detecting class 0: player)

## Challenges

* No ground truth for actual ID labels
* Viewpoints differ significantly, making appearance-based matching preferable
* Lack of jersey number, pose, or embedding-based info limits robustness

## Future Work

* Use appearance embeddings or jersey OCR for better re-identification
* Extend from 1-frame mapping to full video tracking with DeepSORT/ByteTrack
* Add performance metrics for evaluation

## eliverables

* Source Code: `main.py`, `matcher.py`
* YOLO Model: `models/best.pt`
* Sample Videos: `broadcast.mp4`, `tacticam.mp4`
* Output Videos: Annotated views with consistent IDs
* Documentation: `README.md` and `report.md`
