import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from .train_efficientnet import DeepfakeEfficientNet
from .train_swin import DeepfakeSwin
import tempfile
from django.conf import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Constants
MODEL_IMAGE_SIZES = {
    "efficientnet": 300,
    "swin": 224
}

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        model_dir = Path(settings.BASE_DIR) / 'models'
        
        try:
            # Load EfficientNet
            efficientnet_model = DeepfakeEfficientNet()
            efficientnet_model.load_state_dict(
                torch.load(model_dir / 'best_model_efficienet.pth', map_location='cpu')
            )
            efficientnet_model.eval()
            models['efficientnet'] = efficientnet_model

            # Load Swin
            swin_model = DeepfakeSwin()
            swin_model.load_state_dict(
                torch.load(model_dir / 'best_model_swin.pth', map_location='cpu')
            )
            swin_model.eval()
            models['swin'] = swin_model

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

        return models

    def extract_face(self, image, padding=0.1):
        """Extract face from image using MediaPipe"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        configs = [
            {"confidence": 0.5, "model": 1},
            {"confidence": 0.5, "model": 0},
            {"confidence": 0.3, "model": 1},
            {"confidence": 0.3, "model": 0},
            {"confidence": 0.1, "model": 1}
        ]
        
        for config in configs:
            with mp_face_detection.FaceDetection(
                min_detection_confidence=config["confidence"],
                model_selection=config["model"]
            ) as face_detection:
                results = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    
                    pad_w = max(int(bbox.width * width * padding), 0)
                    pad_h = max(int(bbox.height * height * padding), 0)
                    
                    x = max(0, int(bbox.xmin * width) - pad_w)
                    y = max(0, int(bbox.ymin * height) - pad_h)
                    w = min(int(bbox.width * width) + (2 * pad_w), width - x)
                    h = min(int(bbox.height * height) + (2 * pad_h), height - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    face_region = img_cv[y:y+h, x:x+w]
                    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_region_rgb)
                    
                    # Create visualization
                    img_cv_viz = img_cv.copy()
                    cv2.rectangle(img_cv_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    img_viz = cv2.cvtColor(img_cv_viz, cv2.COLOR_BGR2RGB)
                    
                    return face_pil, Image.fromarray(img_viz)
        
        return None, None

    def process_image(self, image, model_type):
        """Process image for model inference"""
        try:
            if image is None:
                return None
                
            img_size = MODEL_IMAGE_SIZES.get(model_type, 224)
            
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            transformed_image = transform(image).unsqueeze(0)
            return transformed_image
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    def analyze_image(self, image_path):
        """Analyze a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            face_image, viz_image = self.extract_face(image)
            
            if face_image is None:
                logger.warning("No face detected in the image.")
                return {
                    'results': [],
                    'error': "No face detected in the image.",
                    'visualization_path': None
                }
            
            results = []
            for model_type, model in self.models.items():
                processed_image = self.process_image(face_image, model_type)
                if processed_image is not None:
                    with torch.no_grad():
                        output = model(processed_image)
                        probability = torch.sigmoid(output).item()
                        prediction = "fake" if probability > 0.5 else "real"
                        confidence = probability if prediction == "fake" else 1 - probability
                        
                        results.append({
                            'model_type': model_type,
                            'prediction': prediction,
                            'confidence': confidence
                        })
            
            if not results:
                logger.warning("No valid predictions from any model.")
                return {
                    'results': [],
                    'error': "Failed to process image with any model.",
                    'visualization_path': None
                }
            
            # Save visualization
            viz_path = os.path.join(settings.MEDIA_ROOT, 'processed', 
                                  f'viz_{os.path.basename(image_path)}')
            viz_image.save(viz_path)
            
            return {
                'results': results,
                'visualization_path': os.path.relpath(viz_path, settings.MEDIA_ROOT),
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing image: {error_msg}")
            return {
                'results': [],
                'error': f"Error analyzing image: {error_msg}",
                'visualization_path': None
            }

    def analyze_video(self, video_path, num_frames=100):
        """Analyze video frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Failed to open video file.")
                return {
                    'results': [],
                    'frame_results': [],
                    'total_frames_analyzed': 0,
                    'error': "Failed to open video file."
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // num_frames)
            
            frames = []
            frame_images = []
            frame_count = 0
            
            # Create directory for frame images
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames_dir = os.path.join('frames', video_id)
            frames_path = os.path.join(settings.MEDIA_ROOT, frames_dir)
            os.makedirs(frames_path, exist_ok=True)
            
            while cap.isOpened() and len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    face_image, _ = self.extract_face(frame_pil)
                    
                    if face_image is not None:
                        # Save the frame image
                        frame_filename = f'frame_{len(frames)}.jpg'
                        frame_path = os.path.join(frames_path, frame_filename)
                        face_image.save(frame_path)
                        frame_images.append(os.path.join(frames_dir, frame_filename))
                        frames.append(face_image)
                        
                frame_count += 1
            
            cap.release()
            
            if not frames:
                logger.warning("No faces detected in video frames.")
                return {
                    'results': [],
                    'frame_results': [],
                    'total_frames_analyzed': 0,
                    'frame_images': [],
                    'error': "No faces detected in video frames."
                }
            
            results = []
            frame_results = []
            
            for model_type, model in self.models.items():
                model_predictions = []
                
                for face in frames:
                    processed_image = self.process_image(face, model_type)
                    if processed_image is not None:
                        with torch.no_grad():
                            output = model(processed_image)
                            probability = torch.sigmoid(output).item()
                            prediction = "fake" if probability > 0.5 else "real"
                            confidence = probability if prediction == "fake" else 1 - probability
                            
                            model_predictions.append({
                                'prediction': prediction,
                                'confidence': confidence
                            })
                
                if model_predictions:
                    fake_count = sum(1 for p in model_predictions if p['prediction'] == "fake")
                    avg_confidence = sum(p['confidence'] for p in model_predictions) / len(model_predictions)
                    fake_ratio = fake_count / len(model_predictions)
                    
                    results.append({
                        'model_type': model_type,
                        'prediction': "fake" if fake_ratio > 0.5 else "real",
                        'confidence': avg_confidence,
                        'fake_frame_ratio': fake_ratio
                    })
                    
                    frame_results.append(model_predictions)
            
            return {
                'results': results,
                'frame_results': frame_results,
                'total_frames_analyzed': len(frames),
                'frame_images': frame_images,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing video: {error_msg}")
            return {
                'results': [],
                'frame_results': [],
                'total_frames_analyzed': 0,
                'frame_images': [],
                'error': f"Error analyzing video: {error_msg}"
            }
