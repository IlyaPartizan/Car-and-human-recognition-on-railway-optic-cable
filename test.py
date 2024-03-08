import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from random import shuffle


def image_in_np(image_name, path):
    img_set = []
    for index in range(len(image_name)):
        img = np.asarray(Image.open(f'{path}/{image_name[index]}').convert('RGB'))
        img_set.append(img)
    return img_set


def decode_output(output):
    for index in range(len(output)):
        output[index] = np.argmax(output[index])
    output = output.tolist()
    new_massive = []
    for index in range(len(output)):
        new_massive.append(int(output[index][0]))
    return new_massive


def check(mas, answer, y):
    new_answer = ''
    if answer == 0:
        new_answer ='car'
    if answer == 1:
        new_answer ='man'
    y_new = ''
    if y == 0:
        y_new ='car'
    if y == 1:
        y_new ='man'
    plt.figure()
    plt.imshow(mas)
    plt.colorbar()
    plt.title(f'Предсказние: {str(new_answer)} Метка: {str(y_new)}')
    plt.grid(False)
    plt.show()

matplotlib.use('TkAgg')
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

model = load_model('model.h5')

output_neuronet = model.predict(x_test)
output_neuronet = decode_output(output_neuronet)

count = 0
for index in range(len(y_test)):
    if output_neuronet[index] == y_test[index]:
        count += 1

print("Количество тестовых данных:", len(y_test))
print("Количество правильных предсказаний:", count)

check(x_test.tolist()[8], output_neuronet[8], y_test[8])
check(x_test.tolist()[9], output_neuronet[9], y_test[9])
check(x_test.tolist()[10], output_neuronet[10], y_test[10])
check(x_test.tolist()[11], output_neuronet[11], y_test[11])