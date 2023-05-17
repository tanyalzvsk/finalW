import os
from os.path import join, dirname, realpath
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from flask import Flask, request, flash, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import numpy as np

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import shutil

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def label_from_folder_map(class_to_label_map):
    return lambda o: class_to_label_map[(o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = '/Users/tanyalzvsk/PycharmProjects/finalW/food101'
    path_img = path + '/images'
    path_meta = path + '/meta/meta'

    train_dir = path + '/train'
    validation_dir = path + '/test'

    BATCH_SIZE = 100  # количество тренировочных изображений для обработки перед обновлением параметров модели
    IMG_SHAPE = 150  # размерность к которой будет преведено входное изображение
    IMG_SIZE = (150, 150)
    UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

     # def model():
     # train_image_generator = ImageDataGenerator(rescale=1. / 255)
     # validation_image_generator = ImageDataGenerator(rescale=1. / 255)

     # train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
     #  directory=train_dir,
     #  shuffle=True,
     #  target_size=(IMG_SHAPE, IMG_SHAPE),
     #  class_mode='sparse')

    # val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #  directory=validation_dir,
    #  shuffle=False,
    #  target_size=(IMG_SHAPE, IMG_SHAPE),
    #  class_mode='sparse')
    # sample_training_images, _ = next(train_data_gen)
    # plotImages(sample_training_images[:5])  # отрисовываем изображения 0-4

   #  model = tf.keras.models.Sequential([
     # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    #  tf.keras.layers.MaxPooling2D(2, 2),

    #  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #  tf.keras.layers.MaxPooling2D(2, 2),

    #  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #  tf.keras.layers.MaxPooling2D(2, 2),

    #  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #  tf.keras.layers.MaxPooling2D(2, 2),

    #  tf.keras.layers.Flatten(),
    #  tf.keras.layers.Dense(128, activation='relu'),
    #  tf.keras.layers.Dense(1, activation='sigmoid'),
    #  tf.keras.layers.Dense(101, activation='softmax')

   #  ])

    #  model.compile(optimizer='adam',
    #  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #  loss='sparse_categorical_crossentropy',
    #  metrics=['accuracy'])

    # EPOCHS = 30
    #  total_train = 75750
    # total_val = 25250
    # history = model.fit(
    # train_data_gen,
    # steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    # epochs=EPOCHS,
    # validation_data=val_data_gen,
    # validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    # )

   # def divideDirectories():
        # f = open(path_meta + '\\classes.txt', 'r')
        # l = [line.strip() for line in f]
        # os.mkdir(path + "\\train\\")
        # for line in l:
        # os.mkdir(path + "\\train\\" + line)

        # f = open(path_meta + '\\train.txt', 'r')
        # l = [line.strip() for line in f]
        # for line in l:
        # s = re.split('/', line)
        # name = path_img + "\\" + s[0] + "\\" + s[1] + '.jpg'
        # shutil.copy2(name, path + "\\train\\" + s[0] + "\\" + s[1] + ".jpg")

        # f = open(path_meta + '\\classes.txt', 'r')
        # l = [line.strip() for line in f]
        # os.mkdir(path + "\\test\\")
        # for line in l:
        # os.mkdir(path + "\\test\\" + line)
        # f = open(path_meta + '\\test.txt', 'r')
        # l = [line.strip() for line in f]

        # for line in l:
        # s = re.split('/', line)
        # name = path_img + "\\" + s[0] + "\\" + s[1] + '.jpg'
        # shutil.copy2(name, path + "\\test\\" + s[0] + "\\" + s[1] + ".jpg")

    # train_images = os.listdir(path_train)
    # for img in train_images:
    # train_labels.append(d[img])

    # print(train_labels[0])
    # print(train_images[0])

    #  model.save_weights('/Users/tanyalzvsk/PycharmProjects/finalW/my_model_weights.h5')
    #  model.save('/Users/tanyalzvsk/PycharmProjects/finalW/my_model.h5')

    model = load_model('my_model.h5')

    #  image = load_img('/Users/tanyalzvsk/PycharmProjects/finalW/food101/train/cheesecake/23673.jpg', target_size=(IMG_SHAPE, IMG_SHAPE))
    #  image = img_to_array(image)
    #  image = np.expand_dims(image, axis=0)

    # prediction = model.predict(image)

    sampleimgpath = '/Users/tanyalzvsk/PycharmProjects/finalW/food101/test/'
    savedimgpath = '/Users/tanyalzvsk/PycharmProjects/finalW/uploads/'
    caloriespath = '/Users/tanyalzvsk/PycharmProjects/finalW/calories.txt'
    subfolders = os.listdir(sampleimgpath)
    class_names = sorted(subfolders)
    class_names.remove('.DS_Store')



    def GetFoodName(imgpath):
        img = tf.keras.preprocessing.image.load_img(imgpath, target_size=(IMG_SHAPE, IMG_SHAPE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        new_img = np.expand_dims(img_array, axis=0)
        prediction = model.predict(new_img / 255.0)
        predicted_class_idx = np.argmax(prediction, axis=-1)[0]
        predicted_class_label = class_names[predicted_class_idx]
        caloriesToReturn = 0
        with open(caloriespath, 'r') as f:
            for line in f:
                name, calories = line.strip().split()
                if name.startswith(predicted_class_label):
                    caloriesToReturn = calories

        print('cs', predicted_class_idx, predicted_class_label)
        return [predicted_class_label, caloriesToReturn]

    @app.route("/file", methods=['POST'])
    @cross_origin()
    def predict():
        image = request.files['image']
        print(image)
        if 'image' not in request.files:
            return 'no image'
        file = request.files['image']
        if file.filename == '':
            return 'no filename'
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        sampleimage = savedimgpath + filename
        [predicted_class_label, calories] = GetFoodName(sampleimage)
        print('data', predicted_class_label, calories)
        return [predicted_class_label, calories]


    app.run(port=8000, debug=True)
