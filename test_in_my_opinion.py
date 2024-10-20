import gensim
import nltk
import os
import gensim.models
from gensim.utils import simple_preprocess
import pymorphy2  
from gensim import corpora 
morph = pymorphy2.MorphAnalyzer()
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from gensim import models
import numpy as np
from nltk.corpus import stopwords   
import gensim.downloader as api
from gensim.models.phrases import Phrases
from gensim.models import FastText
import pandas as pd
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.models
morph = pymorphy2.MorphAnalyzer()
# open the text file as an object
doc = open('taras.txt', encoding ='utf-8')
 
# preprocess the file to get a list of tokens
tokenized =[]
for sentence in doc.read().split('.'):
  # the simple_preprocess function returns a list of each sentence
  tokenized.append(simple_preprocess(sentence, deacc = True))
nltk.download('stopwords')
stop_words = stopwords.words('russian')
Word_lower = []
sen = []
for i in tokenized:
    for word in i:
        a = morph.parse(word)[0].normal_form
        if a not in stop_words:
            sen.append(a)
    Word_lower.append(sen)
    sen =[]

filtered_words = []
for token in Word_lower:
    if token not in stop_words:
        filtered_words.append(token)
model = gensim.models.Word2Vec(sentences=filtered_words, min_count=15, vector_size=500)
model1 = gensim.models.FastText(sentences=filtered_words, min_count=1, vector_size=800)
print(model.wv.most_similar('тарас'))
print('.................................')
print(model1.wv.most_similar('тарас'))
