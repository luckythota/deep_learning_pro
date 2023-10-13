# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:47:21 2023

@author: NAGA LAKSHMI
"""

#to reuse a model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

#loaded_model.save('model_num.hdf5')
#loaded_model=load_model('model_num.hdf5')
#To predict for different data you can use this
import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r"C:\Users\admin\Downloads\0.jpg",target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=loaded_model.predict(test_image)
validation_generator.class_indices
if result[0][0]>=0.5:
    print('dog')
else:
    print('cat')