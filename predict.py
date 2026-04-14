import os
import cv2
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_image
from utils.model_builder import ResNetPreprocess
from utils.gradcam_pp import make_gradcam_plus_plus_heatmap, save_and_display_gradcam_pp
from gradcam.visualize import get_vit_attention_map, save_and_display_gradcam

MODELS_DIR = "models/"
MODELS = {}
CLASS_NAMES = []

def load_all_models():
    global MODELS, CLASS_NAMES
    
    classes_path = os.path.join(MODELS_DIR, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
            
    model_names = ['EfficientNetB0', 'ResNet50', 'MobileNetV2', 'ViT-B16']
    for name in model_names:
        model_path = os.path.join(MODELS_DIR, f"{name}.keras")
        if os.path.exists(model_path):
            try:
                MODELS[name] = tf.keras.models.load_model(model_path, compile=False)
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Warning: Model {name} not found at {model_path}.")

def ensemble_predict(img_path, save_cam_dir="static/gradcam"):
    if not MODELS:
        load_all_models()
        
    img_tensor = preprocess_image(img_path)
    
    predictions = {}
    weighted_sum = np.zeros(len(CLASS_NAMES)) if CLASS_NAMES else np.zeros(1)
    
    weights = {
        'EfficientNetB0': 0.3,
        'ResNet50': 0.2,
        'MobileNetV2': 0.2,
        'ViT-B16': 0.3
    }
    
    total_weight = 0
    generated_cam_path = None
    
    filename = os.path.basename(img_path)
    os.makedirs(save_cam_dir, exist_ok=True)
    
    for name, model in MODELS.items():
        if name in weights:
            pred = model.predict(img_tensor, verbose=0)[0]
            if len(weighted_sum) != len(pred):
                 weighted_sum = np.zeros(len(pred))
            
            predictions[name] = pred
            w = weights[name]
            weighted_sum += pred * w
            total_weight += w
            
            # Generate CAM++ for CNN (Using EfficientNetB0 as the primary visualizer)
            if name == 'EfficientNetB0' and generated_cam_path is None:
                last_conv = "top_activation" # Output spatial map for EB0
                heatmap = make_gradcam_plus_plus_heatmap(img_tensor, model, last_conv)
                cam_path = os.path.join(save_cam_dir, f"cam_pp_{filename}")
                generated_cam_path = save_and_display_gradcam_pp(img_path, heatmap, cam_path)
                
    if total_weight > 0:
        final_probs = weighted_sum / total_weight
    else:
        final_probs = np.zeros(len(CLASS_NAMES))
        
    final_class_idx = np.argmax(final_probs)
    final_class_name = CLASS_NAMES[final_class_idx] if CLASS_NAMES and final_class_idx < len(CLASS_NAMES) else "Unknown"
    confidence = final_probs[final_class_idx] if len(final_probs) > 0 else 0.0
    
    probs_dict = {CLASS_NAMES[i]: float(final_probs[i]) for i in range(len(CLASS_NAMES))} if CLASS_NAMES else {}
    sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)[:5])

    return {
        "disease": final_class_name,
        "confidence": float(confidence),
        "probabilities": sorted_probs,
        "cam_cnn": generated_cam_path.replace("\\", "/") if generated_cam_path else None
    }
