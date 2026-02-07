import os
import cv2
import torch
import tempfile
import traceback
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from torchvision import models, transforms


app = FastAPI(title="Video Moderation API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
AGE_MODEL_PATH = os.path.join(BASE_DIR, "age_group_model_final.pth")


FPS_SAMPLE = 0.5                    # 2 FPS
YOLO_CONF_THRESHOLD = 0.35          # Balanced threshold for weapon detection
WEAPON_BAN_THRESHOLD = 0.7          # Severity threshold for banning

CHILD_RATIO_THRESHOLD = 0.3         # Hạ từ 0.5 xuống 0.3 - 30% frames là trẻ thì detect
MIN_CONSECUTIVE_FACE_FRAMES = 2     # Hạ từ 3 xuống 2
MIN_FACE_AREA_RATIO = 0.05          # Hạ từ 0.08
WEAPON_CONFIDENCE_THRESHOLD = 0.45   # Hạ từ 0.75
CHILD_CONFIDENCE_THRESHOLD = 0.5     # Hạ từ 0.65 xuống 0.5
AGE_CHILD_INDEX = 2 

WEAPON_CLASSES = {
    "gun", "pistol", "rifle",
    "knife", "handgun",
    "firearm", "weapon"
}

device = "cuda" if torch.cuda.is_available() else "cpu"


weapon_model = YOLO(WEAPON_MODEL_PATH)

age_model = models.resnet18(weights=None)
age_model.fc = nn.Sequential(
    nn.Linear(age_model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4)
)
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=device))
age_model.to(device)
age_model.eval()

# ----------------- TRANSFORM -----------------
age_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------- FACE DETECTOR -----------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_frames(video_path: str, fps_sample: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps * fps_sample), 1)

    frames = []
    idx = 0
    timestamp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            frames.append((round(timestamp, 2), frame))
            timestamp += fps_sample

        idx += 1

    cap.release()
    return frames


def has_valid_face(frame) -> bool:
    """Face gate with size + quality constraint"""
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.25,
        minNeighbors=8,
        minSize=(100, 100)
    )

    for (x, y, fw, fh) in faces:
        if (fw * fh) / frame_area >= MIN_FACE_AREA_RATIO:
            return True

    return False


def detect_child(frame) -> tuple:
    """Returns (is_child, confidence)"""
    img = age_transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = age_model(img)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0, pred])
    
    return (pred == AGE_CHILD_INDEX, confidence)


def detect_weapons(frame):
    detections = []
    results = weapon_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)

    for r in results:
        for box in r.boxes:
            label = weapon_model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])

            # Only include high-confidence weapons
            if label in WEAPON_CLASSES and conf >= WEAPON_CONFIDENCE_THRESHOLD:
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2)
                })

    return detections


def infer_weapon_severity(conf: float) -> str:
    return "banned" if conf >= WEAPON_BAN_THRESHOLD else "warning"



@app.post("/moderate-video")
async def moderate_video(file: UploadFile = File(...), debug: bool = False):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    temp_video_path = None

    # Counters
    face_frames = 0
    child_frames = 0
    consecutive_face = 0
    child_timestamps = []
    weapon_events = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_video_path = tmp.name
            tmp.write(await file.read())

        frames = extract_frames(temp_video_path, FPS_SAMPLE)

        # Update main detection loop (replace line 176-210)
        for ts, frame in frames:
            # Frame-level data
            has_face_this_frame = has_valid_face(frame)
            
            if has_face_this_frame:
                consecutive_face += 1
                face_frames += 1
            else:
                consecutive_face = 0

            # Age inference only after stable face detection
            child_in_frame = False
            if consecutive_face >= MIN_CONSECUTIVE_FACE_FRAMES:
                is_child, conf = detect_child(frame)
                # Debug: riêng log để test
                # print(f"[DEBUG] Frame {ts}s: is_child={is_child}, conf={conf:.3f}")
                if is_child and conf > CHILD_CONFIDENCE_THRESHOLD:
                    child_frames += 1
                    child_timestamps.append(ts)
                    child_in_frame = True

            # Weapon detection
            weapons = detect_weapons(frame)
            
            # Only log weapons if no child detected in this frame
            for w in weapons:
                if not child_in_frame:  # Key fix: correlate spatially
                    weapon_events.append({
                        "timestamp": ts,
                        "label": w["label"],
                        "confidence": w["confidence"],
                        "has_child": False
                    })

        child_ratio = (
            child_frames / face_frames
            if face_frames >= MIN_CONSECUTIVE_FACE_FRAMES
            else 0.0
        )

        child_detected = child_ratio >= CHILD_RATIO_THRESHOLD

        flags = set()
        details = []

        # -------- CHILD RESULT --------
        if child_detected:
            flags.add("child")
            details.append({
                "type": "child",
                "source": "custom_age_model",
                "ratio": round(child_ratio, 2),
                "timestamps": child_timestamps[:5]
            })

        # -------- WEAPON RESULT --------
        for w in weapon_events:
            # Child override: completely skip weapons when child detected (likely false positive)
            if child_detected:
                continue

            # Only report high-confidence weapons
            if w["confidence"] < WEAPON_BAN_THRESHOLD:
                continue

            severity = infer_weapon_severity(w["confidence"])
            if severity == "banned":
                flags.add("weapon")

            details.append({
                "type": "weapon",
                "label": w["label"],
                "severity": severity,
                "confidence": w["confidence"],
                "timestamp": w["timestamp"]
            })

        is_safe = len(flags - {"child"}) == 0

        # Debug information to help diagnose empty/incorrect results
        debug_info = {
            "face_frames": face_frames,
            "child_frames": child_frames,
            "consecutive_face": consecutive_face,
            "child_ratio": round(child_ratio, 3),
            "total_frames_sampled": len(frames),
            "weapon_events_count": len(weapon_events)
        }

        resp = {
            "filename": file.filename,
            "is_safe": is_safe,
            "flags": list(flags),
            "details": details
        }

        if debug:
            resp["debug"] = debug_info

        return resp

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
