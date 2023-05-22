# https://medium.com/analytics-vidhya/a-simple-chatbot-using-python-and-nltk-c413b40e9441

import csv
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

greet_in = (
    'hey', 'sup', 'waddup', 'wassup', 'hi', 'hello', 'good day', 'ola', 'bonjour', 'namastay', 'hola', 'heya', 'hiya',
    'howdy',
    'greetings', 'yo', 'ahoy')
greet_out = ['hey', 'hello', 'hi there', 'hi', 'heya', 'hiya', 'howdy', 'greetings', '*nods*', 'ola', 'bonjour',
             'namastay']


def greeting(sent):
    for word in sent.split():
        if word.lower() in greet_in:
            return random.choice(greet_out)


small_talk_responses = {
    'how are you': 'I am fine. Thankyou for asking ',
    'how are you doing': 'I am fine. Thankyou for asking ',
    'how do you do': 'I am great. Thanks for asking ',
    'how are you holding up': 'I am fine. Thankyou for asking ',
    'how is it going': 'It is going great. Thankyou for asking ',
    'goodmorning': 'Good Morning ',
    'goodafternoon': 'Good Afternoon ',
    'goodevening': 'Good Evening ',
    'good day': 'Good day to you too ',
    'whats up': 'The sky ',
    'sup': 'The sky ',
    'thanks': 'Dont mention it. You are welcome ',
    'thankyou': 'Dont mention it. You are welcome ',
    'thank you': 'Dont mention it. You are welcome '
}
small_talk = small_talk_responses.values()
small_talk = [str(item) for item in small_talk]

def tfidf_cosim_smalltalk(doc, query):
    query = [query]
    tf = TfidfVectorizer(use_idf=True, sublinear_tf=True)
    tf_doc = tf.fit_transform(doc)
    tf_query = tf.transform(query)
    cosineSimilarities = cosine_similarity(tf_doc, tf_query).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
    if cosineSimilarities[related_docs_indices] > 0.7:
        ans = [small_talk[i] for i in related_docs_indices[:1]]
        return ans[0]


def naming(name):
    a = name.split()
    if ('my name is' in name):
        for j in a:
            if (j != 'my' and j != 'name' and j != 'is'):
                return j
    elif ('call me' in name):
        for j in a:
            if (j != 'call' and j != 'me'):
                return j
    elif ('name is' in name):
        for j in a:
            if (j != 'name' and j != 'is'):
                return j
    elif ('change my name to' in name):
        for j in a:
            if (j != 'change' and j != 'my' and j != 'name' and j != 'to'):
                return j
    elif ('change name to' in name):
        for j in a:
            if (j != 'name' and j != 'name' and j != 'to'):
                return j
    else:
        return name


f = open('./chatbot.csv', 'r', encoding='utf-8')
reader = csv.reader(f)
corpus = {}
for row in reader:
    corpus[row[0]] = {row[1]: row[2]}

all_text = corpus.values()
#print(all_text)
all_text = [str(item) for item in all_text]


def stem_tfidf(doc, query):
    query = [query]
    p_stemmer = PorterStemmer()
    tf = TfidfVectorizer(use_idf=True, sublinear_tf=True, stop_words=stopwords.words('english'))
    stemmed_doc = [p_stemmer.stem(w) for w in doc]
    stemmed_query = [p_stemmer.stem(w) for w in query]
    tf_doc = tf.fit_transform(stemmed_doc)
    tf_query = tf.transform(stemmed_query)
    return tf_doc, tf_query


def cos_sim(a, b):
    cosineSimilarities = cosine_similarity(a, b).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
    if (cosineSimilarities[related_docs_indices] > 0.5):
        ans = [all_text[i] for i in related_docs_indices[:1]]
        for item in ans:
            c, d = item.split(':')
            return d
    else:
        k = 'I am sorry, I cannot help you with this one. Hope to in the future. Cheers :)'
        return k


stop = True
while (stop == True):
    n = input('\nHello, my name is Nate. What is your name? : ')
    n = n.lower()
    name = naming(n)
    newname = ''
    stop1 = True
    while (stop1 == True):
        query = input('\nHi ' + (newname if len(
            newname) != 0 else name) + ', I am Nate. How can I help you? If you want to exit, type Bye. :')
        query = query.lower()
        query = query.strip("!@#$%^&*()<>,;?")
        if (query == 'bye'):
            stop1 = False
            print('\nNate: This is Nate signing off. Bye, take care' + (newname if len(newname) != 0 else name))

        elif (
                'my name is' in query or 'call me' in query or 'name is' in query or 'change my name to' in query or 'change name to' in query):
            newname = naming(query)
            print('\nNate: Your name is ' + newname)

        elif (
                query == 'what is my name?' or query == 'what is my name' or query == 'whats my name?' or query == 'whats my name'):
            if (len(newname) != 0):
                print('\nNate: Your name is ' + newname)
            else:
                print('\nNate: Your name is ' + name)
        else:
            if (greeting(query) != None):
                print('\nNate: ' + greeting(query) + ' ' + (newname if len(newname) != 0 else name))
            elif (tfidf_cosim_smalltalk(small_talk_responses, query) != None):
                x = tfidf_cosim_smalltalk(small_talk_responses, query)
                print('\nNate: ' + x + (newname if len(newname) != 0 else name))
            else:
                a, b = stem_tfidf(all_text, query)
                g = cos_sim(a, b)
                print('\nNate: ' + g)
    stop = False
