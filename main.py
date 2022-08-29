from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K


K.clear_session()

data_entrenamiento = './data/train'
data_validacion = './data/validation'


epocas = 20 #numero de veces q se va a iterar sobre el set de datos.
width_shape = 224
height_shape = 224 # tamano al cual vamos a procesar las imagenes
batch_size = 32 #cantidad de imagenes que enviamos a procesar en cada uno de los pasos
pasos=200 #numero de veces que se va a procesar la imformacion en cada una de las epocas
pasos_validacion = 200 #
tamano_filtro=(3,3)
tamano_filtro2=(2,2)
clases=2 #tipo de imagenes q vamos a enviar
lr=0.0005 #determina el tamano de los ajustes que ara nuestra red neuronal


## pre procesamiento de imagenes

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=preprocess_input)

valid_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    #save_to_dir='',
    class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(
    data_validacion,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    #save_to_dir='',
    class_mode='categorical')

# Creación y entrenamiento de modelo CNN

nb_train_samples = 10682
nb_validation_samples = 3562

model = Sequential()

inputShape = (height_shape, width_shape, 3)
model.add(Conv2D(32, (3,3), input_shape=inputShape))
model.add(Conv2D(32, (3,3)))
model.add(MaxPool2D())

model.add(Conv2D(64, (3,3)))
model.add(Conv2D(64, (3,3)))
model.add(Conv2D(64, (3,3)))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(clases, activation='softmax', name='output'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model_history = model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)




#Entrenamiento de modelo VGG16

print('#Entrenamiento de modelo VGG16')

image_input = Input(shape=(width_shape, height_shape, 3))

model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

last_layer = model.get_layer('fc2').output
out = Dense(clases, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False

custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

custom_vgg_model.summary()

model_history = custom_vgg_model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)

custom_vgg_model.save("./model/model_VGG16.h5")


print('##Transfer Learning modelo VGG16 - fine tune')

mage_input = Input(shape=(width_shape, height_shape, 3))

model2 = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model2.summary()

last_layer = model2.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(clases, activation='softmax', name='output')(x)
custom_model = Model(image_input, out)
custom_model.summary()

# freeze all the layers except the dense layers
for layer in custom_model.layers[:-3]:
    layer.trainable = False

custom_model.summary()

custom_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model_history = custom_model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)

custom_model.save("./model/model_vgg16_finetune.h5")


print('Transfer Learning modelo Resnet50 - fine tune')

from keras.applications.resnet import ResNet50

image_input = Input(shape=(width_shape, height_shape, 3))

m_Resnet50 = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')

m_Resnet50.summary()

last_layer = m_Resnet50.layers[-1].output

x = Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
out = Dense(clases, activation='softmax', name='output')(x)
custom_model = Model(image_input, out)
custom_model.summary()

# freeze all the layers except the dense layers
for layer in custom_model.layers[:-6]:
    layer.trainable = False

custom_model.summary()

custom_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model_history = custom_model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)

custom_model.save("./model/Resnet50.h5")


print('Transfer Learning modelo VGG19 - fine tune')
from keras.applications.vgg19 import VGG19
image_input = Input(shape=(width_shape, height_shape, 3))
m_VGG19 = VGG19(input_tensor=image_input, include_top=False,weights='imagenet')
m_VGG19.summary()
last_layer = m_VGG19.layers[-1].output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x=Dropout(0.3)(x)
x = Dense(128, activation='relu', name='fc2')(x)
x=Dropout(0.3)(x)
out = Dense(clases, activation='softmax', name='output')(x)
custom_model = Model(image_input, out)
custom_model.summary()

# freeze all the layers except the dense layers
for layer in custom_model.layers[:-6]:
    layer.trainable = False

custom_model.summary()

custom_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model_history = custom_model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)

custom_model.save("./model/VGG19.h5")


print('Modelo Mobilenet - Entrenamiento de toda la red')
from keras.applications.mobilenet import MobileNet


image_input = Input(shape=(width_shape, height_shape, 3))

m_MobileNet = MobileNet(input_tensor=image_input, include_top=False,weights='imagenet')


m_MobileNet.summary()

last_layer = m_MobileNet.layers[-1].output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x=Dropout(0.3)(x)
x = Dense(128, activation='relu', name='fc2')(x)
x=Dropout(0.3)(x)
out = Dense(clases, activation='softmax', name='output')(x)
custom_model = Model(image_input, out)
custom_model.summary()
custom_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model_history = custom_model.fit_generator(
    train_generator,
    epochs=epocas,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    validation_steps=nb_validation_samples//batch_size)

custom_model.save("./model/model_Mobilenet.h5")


print ('##Gráficas de entrenamiento y validación (accuracy - loss)')

def plotTraining(hist, epochs, typeData):
    if typeData == "loss":
        plt.figure(1, figsize=(10, 5))
        yc = hist.history['loss']
        xc = range(epochs)
        plt.ylabel('Loss', fontsize=24)
        plt.plot(xc, yc, '-r', label='Loss Training')
    if typeData == "accuracy":
        plt.figure(2, figsize=(10, 5))
        yc = hist.history['accuracy']
        for i in range(0, len(yc)):
            yc[i] = 100 * yc[i]
        xc = range(epochs)
        plt.ylabel('Accuracy (%)', fontsize=24)
        plt.plot(xc, yc, '-r', label='Accuracy Training')
    if typeData == "val_loss":
        plt.figure(1, figsize=(10, 5))
        yc = hist.history['val_loss']
        xc = range(epochs)
        plt.ylabel('Loss', fontsize=24)
        plt.plot(xc, yc, '--b', label='Loss Validate')
    if typeData == "val_accuracy":
        plt.figure(2, figsize=(10, 5))
        yc = hist.history['val_accuracy']
        for i in range(0, len(yc)):
            yc[i] = 100 * yc[i]
        xc = range(epochs)
        plt.ylabel('Accuracy (%)', fontsize=24)
        plt.plot(xc, yc, '--b', label='Training Validate')

    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('legend', fontsize=18)
    plt.legend()
    plt.xlabel('Number of Epochs', fontsize=24)
    plt.grid(True)

plotTraining(model_history,epocas,"loss")
plotTraining(model_history,epocas,"accuracy")
plotTraining(model_history,epocas,"val_loss")
plotTraining(model_history,epocas,"val_accuracy")