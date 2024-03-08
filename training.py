import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from random import shuffle


def image_in_np(image_name, path):
    img_set = []
    for index in range(len(image_name)):
        img = np.asarray(Image.open(f'{path}/{image_name[index]}').convert('RGB'))
        img_set.append(img)
    return img_set


car_names = os.listdir('car')
man_names = os.listdir('man')
all_names = os.listdir('all')
datas = image_in_np(all_names, 'all')
metrics = []

for name in all_names:
    if name in car_names:
        metrics.append(0)
    elif name in man_names:
        metrics.append(1)

snuffle_mas = []

for index in range(len(metrics)):
    dat = datas[index]
    met = metrics[index]
    snuffle_mas.append([dat, met])

for index in range(10):
    shuffle(snuffle_mas)

for index in range(len(snuffle_mas)):
    datas[index] = snuffle_mas[index][0]
    metrics[index] = snuffle_mas[index][1]

x_train = [datas[index] for index in range(70)]
y_train = [metrics[index] for index in range(70)]
x_test = [datas[index] for index in range(70, len(datas))]
y_test = [metrics[index] for index in range(70, len(metrics))]

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

x_train = np.array(x_train)
x_test = np.array(x_test)

model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train_encoded, validation_data= (x_test, y_test_encoded), epochs=10)
print(hist.history)
model.save('model_test.h5')

matplotlib.use('TkAgg')

plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Точность модели")
plt.ylabel("Точночть")
plt.xlabel("Эпохи")
plt.legend(["учебные", "тестовые"], loc = "upper left")
plt.show()

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Потери модели")
plt.ylabel("Потери")
plt.xlabel("Эпохи")
plt.legend(["учебные", "тестовые"], loc = "upper left")
plt.show()

model.save('model.h5')