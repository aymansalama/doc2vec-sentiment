# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  LinearSVC

import logging
import sys
import nltk
import numpy as np
nltk.download('punkt')

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)



log.info('D2V')
# Loading doc2vec model

model = Doc2Vec.load('./imdb.d2v')
# input sentence
inputS = "I love  she is very beautiful  and smiley "
# tokeniation of the input sentence
tokens = inputS.split()
#
# get the vector from the model
new_vector = model.infer_vector(tokens)
# print(new_vector)
#
sims = model.docvecs.most_similar([new_vector])
print(sims[0])



with open('./input-predict.txt') as f:
    for line in f:
        tokens = line.split()
        new_vector = model.infer_vector(tokens)
        sims = model.docvecs.most_similar([new_vector])
        print('################')
        print(line)
        print(sims[0])
        print('****************')


