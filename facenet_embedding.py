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
  (height, width, channels)

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, channels)
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
  chunk = 10000
  data_dict = dict()
  expand_factor = 0.6
  root_path = '/home/MIBS'
  imgs_folder = os.path.join(root_path, 'test_images')
  model_path = os.path.join(root_path, 'facenet', 'models', 'facenet_keras.h5')
  csv_folder = os.path.join(root_path, 'facenet', 'embedding', str(expand_factor))
  if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)
  error_path = os.path.join(csv_folder, 'error_log.csv')
  model = load_model(model_path)
  model.summary()

  face_detector = dlib.get_frontal_face_detector()
  img_list = os.listdir(imgs_folder)
  error_imgs = []
  chunk_cnt = 1
  for idx, img_name in enumerate(img_list):
    print('Running {}/{}'.format(idx + 1, len(img_list)))
    img_path = os.path.join(imgs_folder, img_name)
    try:
      img = load_image_into_numpy_array(img_path)    
      rects = face_detector(img)
      crop_img = crop_images(img, rects[0], expand=expand_factor)
      result = model(crop_img)
      data_dict[img_name] = np.array(result[0])
      if len(data_dict.keys()) % chunk == 0 or idx == len(img_list) - 1:
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(csv_folder, 'database_embedding_{}.csv'.format(chunk_cnt)), index=False)
        chunk_cnt += 1
        data_dict = dict()
    except:
      print('Error with image' + img_name)
      error_imgs.append(img_name)

  if len(error_imgs) > 0:
    df_e = pd.DataFrame(error_imgs, columns=['Name'])
    df_e.to_csv(error_path, index=False)

if __name__ == "__main__":
    main()