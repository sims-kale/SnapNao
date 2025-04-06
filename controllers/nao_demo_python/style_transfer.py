# style_transfer.py
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load Google's fast style transfer model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def apply_style(content_image, style_image_path='vangogh.jpg'):
    # 1. Load style image
    style_image = tf.io.read_file(style_image_path)
    style_image = tf.image.decode_image(style_image, channels=3)
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    style_image = style_image[tf.newaxis, :]  # Add batch dimension
    
    # 2. Preprocess Webots camera image
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    content_image = content_image[tf.newaxis, :]  # Add batch dimension
    
    # 3. Apply style transfer
    stylized_image = model(tf.constant(content_image), 
                          tf.constant(style_image))[0]
    
    # 4. Convert to Webots-friendly format
    output_image = np.array(stylized_image[0] * 255, dtype=np.uint8)
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)