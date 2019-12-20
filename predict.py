#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--image_dir', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)
parser.add_argument('--checkpoint', help='Point to checkpoint file as str.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--category_names' , default = 'label_map.json', help = 'Mapping of categories to real names.', type = str)
commands = parser.parse_args()
image_path = commands.image_dir
export_path_keras = commands.checkpoint
classes = commands.category_names
top_k = commands.top_k
reloaded = tf.keras.models.load_model(export_path_keras, 
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})
# Create the process_image function
with open('classes', 'r') as f:
    class_names = json.load(f)

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    print("These are the top propabilities",top_values.numpy()[0])
    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes
probs, classes = predict(image, reloaded, top_k)