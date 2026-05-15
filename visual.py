import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

#LOAD MODELS

print("Loading models...")

yolo_model = YOLO("yolov8n.pt")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()


# NORMALIZE

def normalize(x):
    x = np.array(x)

    return x / np.linalg.norm(x)

# IMAGE EMBEDDING

def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        features = clip_model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    features = features[0]
    return features.cpu().numpy()

# EXTRACT FRAMES

video_path = "video.mp4"
frame_folder = "frames"

os.makedirs(frame_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

saved_frames = []

print("Extracting frames...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % 30 == 0:

        filename = f"{frame_folder}/frame_{frame_count}.jpg"

        cv2.imwrite(filename, frame)

        saved_frames.append(filename)

    frame_count += 1

cap.release()

print("Frames extracted!")


# OBJECT DATABASE

print("Creating object database...")

object_embeddings = []
object_images = []

for frame_path in saved_frames:

    results = yolo_model(frame_path)[0]

    image = cv2.imread(frame_path)

    for box in results.boxes:

        cls_id = int(box.cls[0])

        confidence = float(box.conf[0])

        # OPTIONAL FILTER
        if confidence < 0.5:
            continue

        # Bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Crop object
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Convert to PIL
        crop_pil = Image.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )

        # Embedding
        embedding = get_image_embedding(crop_pil)

        embedding = normalize(embedding)

        object_embeddings.append(embedding)

        object_images.append(crop_pil)

print("Objects stored:", len(object_embeddings))


# TEXT SEARCH

def search_text(query):

    valid_queries = [
    "person",
    "helmet"
    "car",
    "bus",
    "truck",
    "bike",
    "motorcycle",
    "bicycle"
]

    if query.lower() not in valid_queries:
        return []

    # -----------------------------
    # TEXT EMBEDDING
    # -----------------------------
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    query_embedding = text_features[0].cpu().numpy()

    query_embedding = np.array(query_embedding).reshape(-1)

    query_embedding = normalize(query_embedding)

    # -----------------------------
    # SIMILARITY SEARCH
    # -----------------------------
    scores = []

    for i, emb in enumerate(object_embeddings):

        emb = np.array(emb).reshape(-1)

    # MATCH VECTOR SIZE
        min_len = min(
            len(query_embedding),
            len(emb)
        )

        q = query_embedding[:min_len]

        e = emb[:min_len]

        score = float(np.dot(q, e))

        if score > 0.15:
            scores.append((object_images[i], score))

        if len(scores) == 0:
            return []
        
    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:5]

# IMAGE SEARCH FUNCTION


def search_image(query_image):


    # QUERY IMAGE EMBEDDING
  
    query_embedding = get_image_embedding(query_image)
    query_embedding = query_embedding.reshape(-1)

    query_embedding = normalize(query_embedding)

    # SIMILARITY SEARCH
   
    scores = []

    for i, emb in enumerate(object_embeddings):
        emb = emb.reshape(-1)

        score = float(
            np.dot(query_embedding, emb)
        )

        if score > 0.15:
            scores.append((object_images[i], score))

        if len(scores) == 0:
            return []

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:5]


  