import pandas as pd
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.models
morph = pymorphy2.MorphAnalyzer()
from gensim.models import FastText

nltk.download('punkt')
bank = pd.read_excel('vullist.xlsx')
text = bank.iloc[0]['Наименование уязвимости']
tokens = text.lower().split()
marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...']
only_words = []
for token in tokens:
    if token not in marks:
        if token[-1] in marks:
            token = token[:(len(token)-1)]
        only_words.append(token)
lem = []
for token in only_words:
    lem.append(morph.parse(token)[0].normal_form)
nltk.download('stopwords')
stop_words = stopwords.words('russian')
filtered_words = []
for token in lem:
    if token not in stop_words:
        filtered_words.append(token)

def preprocess(text, stop_words, marks, morph):
    tokens = str(text).lower().split()
    preprocess_text = []
    for token in tokens:
        if token not in marks:
            if token[-1] in marks:
                token = token[:(len(token))]
                lem = morph.parse(token)[0].normal_form
                if lem not in stop_words:
                    preprocess_text.append(lem)
    return preprocess_text
bank['Preprocessing_text_НАИМЕНОВАНИЕ_УБИ'] = bank.apply(lambda row: preprocess(row['Наименование уязвимости'], marks, stop_words, morph), axis=1)
bank['Preprocessing_text_Описание'] = bank.apply(lambda row: preprocess(row['Описание уязвимости'], marks, stop_words, morph), axis=1)



A = list(bank['Preprocessing_text_НАИМЕНОВАНИЕ_УБИ'])

A.extend(list(bank['Preprocessing_text_Описание']))

B = [ele for ele in A if ele != []]
model = gensim.models.Word2Vec(sentences=B, min_count=2, vector_size=500)
modelf= gensim.models.FastText(sentences=B, min_count=2, vector_size=500)

print(modelf.wv.most_similar('угроза'))
print('..................................')
print(model.wv.most_similar('угроза'))

