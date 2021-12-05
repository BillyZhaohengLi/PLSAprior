import numpy as np
from sklearn.metrics import confusion_matrix
from FastPlsa import *

## Demo of fastPlsa. The 20newsgroups dataset is required.
## (download the dataset, and run 20news_converter.py first.)

## Create a fastPlsa instance using the test documents and optionally training documents & labels.
## fastPlsa can also be created using an existing saved model (see below).
documents_path_train = '20news_train.txt'
labels_path_train = '20news_label.txt'
documents_path_test = '20news_test.txt'
fastPlsa = FastPlsa(documents_path_test, documents_path_train, labels_path_train)  # instantiate corpus

## If training documents & labels are provided, the number of topics are inferred from the number of unique labels.
## Otherwise, it should be provided explicitly during initialization.
random = True
mixing_weight = 0.7
number_of_topics = 20
fastPlsa.initialize(random, mixing_weight, number_of_topics)

## Run the EM algorithm, autosave every 10 iterations.
iterations = 30
epsilon = 0.001
fastPlsa.plsa(iterations, epsilon, save_every_iter = 10)


## Check model status and save
fastPlsa.show_status()
fastPlsa.save_model()

## Create fastPlsa using an existing saved model
fastPlsa2 = FastPlsa(model_name = 'model')  # instantiate corpus
gt_matrix = []
gt = open("20news_gt.txt", "r")
for line in gt:
    gt_matrix.append(int(line))
fastPlsa2.evaluate_model(np.array(gt_matrix))
