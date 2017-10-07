from pandas import read_csv
from tokenize import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy
import re
import string
import nltk
import csv
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

_regexPunctuation = re.compile('[%s0-9]+' % re.escape(string.punctuation + '~!@#$%^&*():;-_+=\t,\.\\/[]{}"\'|`?'))
_regexRusToken = re.compile('[а-я]+')
_regexToken = re.compile('[a-zа-я]+')

def read_data():
	return read_csv("data/train.csv", encoding = 'utf-8')
	#data['category_id','title','description','price']

def text_tokenize(text):
	text = text.lower()
	text = _regexPunctuation.sub(' ', text).strip()
	tokens = _regexToken.findall(text)
	tokens_res = [t for t in tokens if len(t) > 2]
	tokens_res = tokens_normalize(tokens_res)

def token_text_corp(text_corp):
	text_corp['title'] = text_corp.apply(lambda x: text_tokenize(x['title']), axis=1)
	text_corp['description'] = text_corp.apply(lambda x: text_tokenize(x['description']), axis=1)
	return text_corp

def text_corp_to_csv(text_corp):
	text_corp['title'].to_csv('title_token.csv', sep=',', encoding='utf-8')
	text_corp['description'].to_csv('description_token.csv', sep=',', encoding='utf-8')
	text_corp['price'].to_csv('price_token.csv', sep=',', encoding='utf-8')
	text_corp['category_id'].to_csv('category.csv', sep=',', encoding='utf-8')

def vectorize(text_corpus, u_idf):
	count_vect = CountVectorizer(ngram_range=(1,2))
	X_train_counts = count_vect.fit_transform(text_corpus)
	numpy.savetxt('feature_names.csv', numpy.array(X_train_counts.get_feature_names()))
	tfidf_transformer = TfidfTransformer(use_idf=u_idf)
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	print(X_train_tfidf.shape)
	return X_train_tfidf

def tokens_normalize(tokens):
	result = []
	for t in tokens:
		if (_regexRusToken.search(t)):
			p = morph.parse(t)[0]		
			result.append(p.normal_form)
		else:
			result.append(t)
	return result

data = read_data()
text_corp = token_text_corp(data)
text_corp_to_csv(text_corp)
x_title = vectorize(text_corp['title'], False)
x_description = vectorize(text_corp['description'], True)

numpy.savetxt("x_title.csv", x_title, delimiter=",", encoding='utf-8')
numpy.savetxt("x_description.csv", x_description, delimiter=",", encoding='utf-8')
