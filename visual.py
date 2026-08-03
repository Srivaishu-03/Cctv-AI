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

yolo_class_names = yolo_model.names


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
object_classes = []   # NEW — tracks what YOLO thinks each object actually is


for frame_path in saved_frames:

    results = yolo_model(frame_path)[0]

    image = cv2.imread(frame_path)

    for box in results.boxes:

        cls_id = int(box.cls[0])

        confidence = float(box.conf[0])

        # OPTIONAL FILTER
        if confidence < 0.5:
            continue

         # YOLO's own label for this object, e.g. "person", "truck", "car"
        class_name = yolo_class_names[cls_id]

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
        # Keep track of YOLO's class label for this object (aligns with embeddings/images)
        object_classes.append(class_name.lower())

print("Objects stored:", len(object_embeddings))


# TEXT SEARCH

def search_text(query):

    valid_queries = [
    "person",
    "helmet",
    "car",
    "bus",
    "truck",
    "bike",
    "motorcycle",
    "bicycle",
    "van"
]
    query_lower = query.lower()


    if query.lower() not in valid_queries:
        return []
    
    # step 1 Filter candidates using YOLO's own class label.
    # This guarantees a "person" search can never return a truck, etc.,
    # no matter what CLIP's similarity score says.
    candidate_indices = [
        i for i, cls in enumerate(object_classes) if cls == query_lower
    ]
 
    if len(candidate_indices) == 0:
        return []


    # step 2 TEXT EMBEDDING

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

    
    # Step 3 SIMILARITY SEARCH
    
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
   
    # Use YOLO on the query image to get a class label and filter candidates.
    # This prevents unrelated object types from being returned for an arbitrary
    # query image (e.g., returning cars for a person photo).
    img_bgr = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2BGR)

    yres = yolo_model(img_bgr)[0]

    detected_class = None
    max_conf = 0.0

    for box in yres.boxes:

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf > max_conf:
            max_conf = conf
            detected_class = yolo_class_names[cls_id].lower()

    # Require a reasonably confident detection; otherwise return no results
    # to avoid returning unrelated objects.
    if detected_class is None or max_conf < 0.35:
        return []

    # Filter candidates to only those YOLO labeled with the same class
    candidate_indices = [i for i, cls in enumerate(object_classes) if cls == detected_class]

    if len(candidate_indices) == 0:
        return []

    # Higher similarity threshold for image->image matches
    SIM_THRESHOLD = 0.25

    scores = []

    for i in candidate_indices:
        emb = object_embeddings[i].reshape(-1)

        score = float(np.dot(query_embedding, emb))

        if score > SIM_THRESHOLD:
            scores.append((object_images[i], score))

    if len(scores) == 0:
        return []

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:5]


  