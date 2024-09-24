# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 16:47:18 2023

@author: Divya
"""

import numpy as np
import pickle


# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\Divya\OneDrive\Documents\mini project\trained_model.sav','rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
