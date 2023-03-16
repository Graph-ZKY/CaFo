# -*- coding: utf-8 -*-
"""
This part of the program can separate images that are placed together into training, testing, and validation sets.
However, there is an issue that this dataset is extracted from ImageNet, and the categories of the training,
testing, and validation sets are different. The total number of categories is 100.
This means that we cannot train from scratch. Therefore, it is necessary to redivide this dataset.
"""

import csv
import os
from PIL import Image
import os
import random
import shutil


def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def moveFile(file_dir, save_dir):
    ensure_dir_exists(save_dir)
    path_dir = os.listdir(file_dir)
    filenumber = len(path_dir)
    rate = 0.1667  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)
    print(picknumber)
    sample = random.sample(path_dir, picknumber)
    # print (sample)
    for name in sample:
        shutil.move(file_dir + name, save_dir + name)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

train_csv_path = "./mini_imagenet/train.csv"
val_csv_path = "./mini_imagenet/val.csv"
test_csv_path = "./mini_imagenet/test.csv"

train_label = {}
val_label = {}
test_label = {}
with open(train_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(val_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

img_path = "./mini_imagenet/images"
new_img_path = "./mini_imagenet/images_OK"
for png in os.listdir(img_path):
    path = img_path + '/' + png
    im = Image.open(path)
    if (png in train_label.keys()):
        tmp = train_label[png]
        temp_path = new_img_path + '/train' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)
        # with open(temp_path, 'wb') as f:
        #     f.write(path)

    elif (png in val_label.keys()):
        tmp = val_label[png]
        temp_path = new_img_path + '/val' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)

    elif (png in test_label.keys()):
        tmp = test_label[png]
        temp_path = new_img_path + '/test' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)



os.mkdir('./mini_imagenet/images_CDD')
os.mkdir('./mini_imagenet/images_CDD/data')


for file in os.listdir('./mini_imagenet/images_OK/train') :
    shutil.move('./mini_imagenet/images_OK/train/'+file, './mini_imagenet/images_CDD/data')
for file in os.listdir('./mini_imagenet/images_OK/val') :
    shutil.move('./mini_imagenet/images_OK/val/'+file, './mini_imagenet/images_CDD/data')
for file in os.listdir('./mini_imagenet/images_OK/test') :
    shutil.move('./mini_imagenet/images_OK/test/'+file, './mini_imagenet/images_CDD/data')




"""
This code will extract a certain proportion (which can be defined by the user, here it is 1/6) 
of data from each category in the "data" dataset and reorganize them into a test set 
(note that this is not a copy, but rather a cut).
"""

path = './mini_imagenet/images_CDD/'
dirs = os.listdir(path + 'data/')
for file in dirs:
    file_dir = path + 'data/' + file + '/'
    print(file_dir)
    save_dir = path + 'test/' + file
    print(save_dir)
    mkdir(save_dir)
    save_dir = save_dir + '/'
    moveFile(file_dir, save_dir)
os.rename(path+'data',path+'train')

