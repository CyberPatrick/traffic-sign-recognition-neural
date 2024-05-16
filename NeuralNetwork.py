import numpy as np

from skimage import color, exposure, transform
from PIL import Image
import io
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 48
NUM_CLASSES = 164
RANDOM_SEED=45

def preprocess_img(img):

    # Histogram normalization in v channel
    try:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
    except:
        print("Error with color")

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2]) - 1

model = Sequential()

model.add(Conv2D(48, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(96, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('model.weights.h5')


def predict(img: bytes):
    ppm_image = np.array(Image.open(io.BytesIO(img)))
    ppm_image = preprocess_img(ppm_image)
    ppm_image = np.expand_dims(ppm_image, axis=0)
    return model.predict(ppm_image, verbose=0)
