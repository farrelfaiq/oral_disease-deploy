import numpy as np
from PIL import Image
import io
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from utils.config import IMAGE_SIZE
from utils.config import IMAGE_SIZE
from utils.logger import logger


# Load ResNet50 
logger.info("Loading EfficientNetB0 model for feature extraction...")
resnet_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")

def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes ke format input EfficientNetB0"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    return x

def extract_features(img_array: np.ndarray) -> np.ndarray:
    """Ekstraksi fitur dari EfficientNetB0"""
    features = resnet_model.predict(img_array)
    return features
