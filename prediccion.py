import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import numpy as np
import cv2

width_shape = 224
height_shape = 224

names = ['MELANOMA', 'NO_MELANOMA']

modelt = load_model("./model/model_Mobilenet.h5")
#modelt = custom_vgg_model

imaget_path = "nomelanoma5.jpg"
imaget=cv2.resize(cv2.imread(imaget_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = modelt.predict(xt)

print(names[np.argmax(preds)])



modelt = load_model("./model/model_vgg16.h5")
#modelt = custom_vgg_model

imaget=cv2.resize(cv2.imread(imaget_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = modelt.predict(xt)
print(names[np.argmax(preds)])


plt.imshow(cv2.cvtColor(np.asarray(imaget),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()