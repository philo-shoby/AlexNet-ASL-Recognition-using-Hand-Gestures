import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import cv2, os

!unzip ".../asl-alphabet.zip"
train_dir = '.../asl_alphabet_train/asl_alphabet_train'
train_folders = os.listdir(train_dir)
test_dir = '.../asl_alphabet_test/asl_alphabet_test'
test_files = os.listdir(test_dir)

x_train, y_train = [], []
for folder in train_folders:
    files = os.listdir(train_dir + '/'+ folder)
    print('Reading images from ' + train_dir + folder + '/ ...')
    for file in files[:1000]:
        img = cv2.imread(train_dir +'/'+ folder + '/' + file)
        img = cv2.resize(img, (227, 227))
        x_train.append(img)
        y_train.append(folder)
x_test, y_test = [], []
for file in test_files:
    img = cv2.imread(test_dir +'/'+ file)
    img = cv2.resize(img, (227, 227))
    x_test.append(img)
    y_test.append(file.split('_')[0])
    
