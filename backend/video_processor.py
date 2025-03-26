import cv2
import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import librosa
import ffmpeg
from transformers import pipeline
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from fpdf import FPDF
import math
from sklearn.cluster import KMeans

# Directories
UPLOAD_FOLDER = "backend/uploads"
FRAMES_FOLDER = "backend/frames"
KEYFRAMES_FOLDER = "backend/keyframes"
COMIC_FOLDER = "backend/generated_comics"
AUDIO_FOLDER = "backend/audio"

os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(KEYFRAMES_FOLDER, exist_ok=True)
os.makedirs(COMIC_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load Pretrained ResNet for Feature Extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Speech Recognition Model
speech_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")


def extract_frames(video_path, output_dir, frame_interval=2):
    """Extract frames at fixed intervals."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()


def compute_brightness(image_path):
    """Compute brightness score of an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.mean(img) / 255.0


def compute_motion(prev_img, curr_img):
    """Compute motion score between two consecutive grayscale frames."""
    diff = cv2.absdiff(prev_img, curr_img)
    return np.mean(diff) / 255.0


def extract_features(image_path):
    """Extract deep features from an image using ResNet."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = resnet(img)
    
    return features.squeeze().cpu().numpy()  # Shape: (2048,)


def detect_scenes(video_path):
    """Detect scene changes using SceneDetect."""
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Convert scene times into frame numbers
    fps = video_manager.get_framerate()
    scenes = [(int(start_time.get_seconds() * fps), int(end_time.get_seconds() * fps)) 
              for start_time, end_time in scene_manager.get_scene_list()]

    return scenes


def compute_highlight_scores(frames_dir):
    """Compute highlight scores for frames using brightness, motion & aesthetics."""
    frame_files = sorted(os.listdir(frames_dir))
    highlight_scores = {}
    prev_frame = None  # For motion computation

    for i, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_name)
        curr_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        # Compute brightness
        brightness_score = compute_brightness(frame_path)

        # Compute motion
        motion_score = compute_motion(prev_frame, curr_frame) if prev_frame is not None else 0
        prev_frame = curr_frame

        # Extract aesthetics (Deep Features from ResNet)
        feature_vector = extract_features(frame_path)
        aesthetic_score = np.linalg.norm(feature_vector) / np.linalg.norm(feature_vector, ord=2)

        # Final highlight score (weighted sum)
        highlight_scores[frame_name] = (0.5 * brightness_score) + (0.3 * motion_score) + (0.2 * aesthetic_score)

    return highlight_scores


def select_keyframes(frames_dir, scenes, num_keyframes=3):
    """Selects best keyframes using highlight scores & clustering."""
    highlight_scores = compute_highlight_scores(frames_dir)
    frame_files = sorted(os.listdir(frames_dir))
    selected_keyframes = []

    for scene in scenes:
        start, end = scene
        scene_frames = frame_files[start:end]

        if len(scene_frames) == 0:
            continue

        # Sort scene frames by highlight score
        sorted_frames = sorted(scene_frames, key=lambda x: highlight_scores.get(x, 0), reverse=True)

        # Extract deep features for clustering
        feature_vectors = np.array([extract_features(os.path.join(frames_dir, f)) for f in sorted_frames])

        # Apply KMeans clustering to get diverse frames
        kmeans = KMeans(n_clusters=min(num_keyframes, len(feature_vectors)), random_state=42)
        kmeans.fit(feature_vectors)
        cluster_centers = kmeans.cluster_centers_

        # Select frames closest to cluster centers
        selected_frames = []
        for center in cluster_centers:
            closest_idx = np.argmin([np.linalg.norm(f - center) for f in feature_vectors])
            selected_frames.append(sorted_frames[closest_idx])

        selected_keyframes.extend(selected_frames)

    return selected_keyframes


def process_video(video_path):
    """Main function to process video into a comic."""
    extract_frames(video_path, FRAMES_FOLDER)
    scenes = detect_scenes(video_path)
    keyframes = select_keyframes(FRAMES_FOLDER, scenes)

    # Extract & Process Audio
    audio_path = os.path.join(AUDIO_FOLDER, "audio.wav")
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)

    # Break transcript into smaller parts
    captions = transcript.split(". ")

    # Apply Style Transfer & Overlay Captions
    for i, keyframe in enumerate(keyframes):
        frame_path = os.path.join(FRAMES_FOLDER, keyframe)
        apply_style_transfer(frame_path)
        overlay_text_on_image(frame_path, captions[i] if i < len(captions) else "No caption")

        # Move processed frame to keyframes folder
        cv2.imwrite(os.path.join(KEYFRAMES_FOLDER, keyframe), cv2.imread(frame_path))

    # Generate Comic PDF
    pdf_name = os.path.basename(video_path).replace(".mp4", ".pdf")
    pdf_path = create_comic_pdf(keyframes, captions, pdf_name)
    return pdf_name
