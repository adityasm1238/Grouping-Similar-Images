from keras.applications import MobileNetV2

from keras.preprocessing import image

from tqdm import tqdm 

from PIL import Image
import numpy as np
import tensorflow as tf
import os
from numpy import savez_compressed


files = os.listdir('data')
enc = np.zeros((len(files),1000))
model = MobileNetV2(classifier_activation=None)
for i in tqdm (range (len(files)), desc="Encoding files..",unit="file"):
	try:
		im = np.asarray(image.load_img('data/'+files[i],target_size=(224,224)))
		im = im[:,:,:3]/225
		e = model.predict(np.expand_dims(im, axis=0))
		enc[i,:]=e[0,:]
	except:
		print("Error reading image:"+files[i])

savez_compressed("data.npz",enc)







