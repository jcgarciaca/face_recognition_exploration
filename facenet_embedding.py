import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
import numpy as np
from PIL import Image
import pandas as pd
import os

IMG_SIZE = (160, 160)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image_np = np.array(Image.open(path))
  return image_np

def crop_images(image, rect, expand=0.4):
    crop_img = Image.fromarray(image.copy())
    v_dist = int(abs(rect.bottom() - rect.top()) * expand / 2)
    h_dist = int(abs(rect.right() - rect.left()) * expand / 2)
    top = max(0, rect.top() - v_dist)
    bottom = min(rect.bottom()+v_dist, image.shape[0])
    left = max(0, rect.left()-h_dist)
    right = min(rect.right()+h_dist, image.shape[1])
    crop_img = crop_img.crop((left, top, right, bottom))
    return np.array(crop_img.resize(IMG_SIZE))[np.newaxis, :, :, :] / 255.


def main():
  data_dict = dict()
  root_path = '/home/MIBS'
  imgs_folder = os.path.join(root_path, 'images')
  model_path = os.path.join(root_path, 'facenet', 'models', 'facenet_keras.h5')
  csv_path = os.path.join(root_path, 'facenet', 'embedding', 'database_embedding.csv')
  model = load_model(model_path)
  model.summary()

  face_detector = dlib.get_frontal_face_detector()
  img_list = os.listdir(imgs_folder)
  for idx, img_name in enumerate(img_list):
    print('Running {}/{}'.format(idx + 1, len(img_list)))
    img_path = os.path.join(imgs_folder, img_name)
    try:
      img = load_image_into_numpy_array(img_path)    
      rects = face_detector(img)
      crop_img = crop_images(img, rects[0], expand=0.2)
      result = model(crop_img)
      data_dict[img_name] = np.array(result[0])
    except:
      print('Error with image' + img_name)

  df = pd.DataFrame.from_dict(data_dict)
  df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()