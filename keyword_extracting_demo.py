import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from lxml import etree
import numpy as np

unwanted = stopwords.words('english') + list(string.punctuation)
root = etree.parse('news.xml').getroot()
corpus = root[0]

all_headlines = []
all_stories = []
all_st_tokens = []  # store each story as a list of tokens


for news in corpus:
    headline = news[0].text
    story = news[1].text.lower()
    all_st_tokens.append(word_tokenize(story))
    all_headlines.append(headline)
    all_stories.append(story)

vocabulary = []
lemmatizer = WordNetLemmatizer()
nn_words = []  # all words from all stories that have a 'NN' tag
                # call fit transform on this list

for list_of_tokens in all_st_tokens:
    current_nn_words = [] 
    for token in list_of_tokens:
        word = lemmatizer.lemmatize(token)
        tag = nltk.pos_tag([word])
        if tag[0][1] == 'NN' and tag[0][0] not in unwanted+vocabulary:
            vocabulary.append(tag[0][0])
        if tag[0][1] == 'NN' and tag[0][0] not in unwanted:
            current_nn_words.append(tag[0][0])
    nn_words.append(' '.join(current_nn_words))

vectorizer = TfidfVectorizer(input='content', use_idf=True,
                             analyzer='word',
                             vocabulary=vocabulary)

matrix = vectorizer.fit_transform(nn_words)
arr = matrix.toarray()

for i, headline in enumerate(all_headlines):
    print(headline + ':')
    top_idx = arr[i].argsort()[::-1]
    value_word = []  # putting them in the form of [tfidf score, word]
# in order to sort by tfidf score first then by alphabetical order
    for idx in top_idx:
        value_word.append([arr[i][idx], vocabulary[idx]])
    words = sorted(value_word, key=lambda x: (x[0], x[1]), reverse=True)
    for idx in range(5):
        print(words[idx][1], end=' ')
    print()





