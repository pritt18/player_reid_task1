import cv2
import os
from ultralytics import YOLO
from utils.matcher import match_players

model_path = 'models/best.pt'
broadcast_path = 'data/broadcast.mp4'
tacticam_path = 'data/tacticam.mp4'
out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

model = YOLO(model_path)

broadcast_cap = cv2.VideoCapture(broadcast_path)
tacticam_cap = cv2.VideoCapture(tacticam_path)

w, h = int(broadcast_cap.get(3)), int(broadcast_cap.get(4))
fps = broadcast_cap.get(cv2.CAP_PROP_FPS)

broadcast_out = cv2.VideoWriter(f'{out_dir}/broadcast_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
tacticam_out = cv2.VideoWriter(f'{out_dir}/tacticam_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

ret1, frame_b = broadcast_cap.read()
ret2, frame_t = tacticam_cap.read()

if not ret1 or not ret2:
    print("Error: Could not read frames from both videos.")
    exit()

results_t = model.predict(source=frame_t, conf=0.5, verbose=False)[0]
results_b = model.predict(source=frame_b, conf=0.5, verbose=False)[0]

players_b = [box.xyxy[0].cpu().numpy() for box in results_b.boxes if int(box.cls[0]) == 0]  # class 0 = player
players_t = [box.xyxy[0].cpu().numpy() for box in results_t.boxes if int(box.cls[0]) == 0]

print(f"[Broadcast] Players detected: {len(players_b)}")
print(f"[Tacticam] Players detected: {len(players_t)}")

matched_ids = match_players(players_b, players_t)

for i, box in enumerate(players_b):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame_b, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_b, f"ID: {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

for i, box in enumerate(players_t):
    x1, y1, x2, y2 = map(int, box)
    id_matched = matched_ids.get(i, "?")
    cv2.rectangle(frame_t, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame_t, f"ID: {id_matched}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

broadcast_out.write(frame_b)
tacticam_out.write(frame_t)

broadcast_cap.release()
tacticam_cap.release()
broadcast_out.release()
tacticam_out.release()

print(" Output videos saved in 'output' folder")
