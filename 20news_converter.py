import os
import re
import numpy as np

# program for converting the 20newsgroup dataset into
# format processable by fastPlsa.
# dataset found at http://qwone.com/~jason/20Newsgroups/
train_path = '20news-bydate/20news-bydate-train'
test_path = '20news-bydate/20news-bydate-test'
train_file = open("20news_train.txt", "w")
label_file = open("20news_label.txt", "w")
test_file = open("20news_test.txt", "w")
gt_file = open("20news_gt.txt", "w")

train_size = 20
test_size = 10

label_count = 0
train_dirs = os.scandir(train_path)
for category in train_dirs:
    if category.is_dir():
        cur_size = train_size
        category_files = os.scandir(train_path + "/" + category.name)
        for item in category_files:
            if item.is_file():
                print(label_count, file = label_file)
                item_handle = open(train_path + "/" + category.name + "/" + item.name, "r", errors = 'ignore')
                contents = re.sub("[^0-9a-zA-Z]+", ' ', item_handle.read())
                print(contents, file = train_file)
                cur_size -= 1
                if cur_size == 0:
                    break
        label_count += 1
train_file.close()
label_file.close()

label_count = 0
test_dirs = os.scandir(test_path)
for category in test_dirs:
    if category.is_dir():
        cur_size = test_size
        category_files = os.scandir(test_path + "/" + category.name)
        for item in category_files:
            if item.is_file():
                print(label_count, file = gt_file)
                item_handle = open(test_path + "/" + category.name + "/" + item.name, "r", errors = 'ignore')
                contents = re.sub("[^0-9a-zA-Z]+", ' ', item_handle.read())
                print(contents, file = test_file)
                cur_size -= 1
                if cur_size == 0:
                    break
        label_count += 1
test_file.close()
gt_file.close()