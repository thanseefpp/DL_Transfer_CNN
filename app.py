import numpy as np
from keras.utils import load_img, img_to_array
import pickle

load_model = pickle.load(open('dataset/DL_cat_or_dog_image_processed.pkl','rb'))

def predict_output():
  test_image = load_img('dataset/sample_data/dog.jpeg',target_size = (64,64))
  test_image = img_to_array(test_image)
  test_image = np.expand_dims(test_image,axis=0)#expanding the array size (batching)
  result = load_model.predict(test_image)

  if result[0][0] == 1:
    print("dog")
  else:
    print('cat')
  
  
if __name__ == '__main__':
  predict_output()