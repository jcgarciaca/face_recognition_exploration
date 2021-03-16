import pandas as pd
import numpy as np
import os


def findCosineSimilarity(source_representation, test_representation):
  a = np.matmul(np.transpose(source_representation), test_representation)
  b = np.sum(np.multiply(source_representation, source_representation))
  c = np.sum(np.multiply(test_representation, test_representation))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


root = '/home/MIBS/facenet/embedding'
src_csv = os.path.join(root, 'database_embedding.csv')
dst_csv = os.path.join(root, 'similarity.csv')
print('File exists: {}'.format(os.path.exists(src_csv)))

df = pd.read_csv(src_csv)
print('Embedding loaded')

data_dict2 = {}
print('Calculating similarity...')
for label_1, embedding_1 in df.items():
  sub_dict = {}
  for label_2, embedding_2 in df.items():
    score = round(findCosineSimilarity(np.array(embedding_1), np.array(embedding_2)), 6)
    sub_dict[label_2] = score
    print('Img A: {}, Img B: {}, Score: {}'.format(label_1, label_2, score))
  data_dict2[label_1] = sub_dict

df2 = pd.DataFrame.from_dict(data_dict2)
df2.to_csv(dst_csv, index=False)