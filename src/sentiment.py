from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import numpy

categories_labels = ["subjective", "objective"]

n_instances = 5000
subjective_documents = subjectivity.sents(categories="subj")[:n_instances]
subjective_categories = numpy.array([0]).repeat(n_instances)

objective_documents = subjectivity.sents(categories="obj")[:n_instances]
objective_categories = numpy.array([1]).repeat(n_instances)

documents = subjective_documents + objective_documents
categories = numpy.concatenate((subjective_categories, objective_categories))

documents_train, documents_test, categories_train, categories_test = train_test_split(documents, categories, stratify=categories)

def extract_frequent_unigrams(documents, min_freq):
    # TODO: stop using nltk to make this more efficient
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(document) for document in documents])
    return sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=min_freq)

class UnigramPresence:
    def __init__(self, min_freq: int):
        self.min_freq = min_freq
        self.unigrams = None

    def transform(self, documents):
        return numpy.array([
            self.transform_single(document) for document in documents
        ])

    def transform_single(self, document):
        # This is equivalent to extract_unigram_feats, except that:
        # - it isn't absurdly inefficient
        # - it returns an array that can be used with sklearn models
        document_set = set(document)
        return numpy.array([(unigram in document_set) for unigram in self.unigrams]).astype(float)

    def fit(self, documents, categories):
        self.unigrams = extract_frequent_unigrams(documents, self.min_freq)
        print("Fitted")
        return self

pipeline = Pipeline([
    ("extract_unigram", UnigramPresence(200)),
    # ("naive_bayesian", CategoricalNB())
    ("mlp_classifier", MLPClassifier([100, 20], max_iter=1000))
])

print("Loaded")
pipeline.fit(documents_train, categories_train)

print(pipeline.score(documents_test, categories_test))
print(objective_documents[3], categories_labels[pipeline.predict([objective_documents[3]])[0]])

for unigram in pipeline["extract_unigram"].unigrams:
    proba = pipeline.predict_proba([unigram])[0]
    print(unigram, categories_labels[numpy.argmax(proba)])
# print(pipeline["extract_unigram"].unigrams)

# subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
# obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

# def split_data(document_lists, test_size):
#     def split(documents):
#         return train_test_split(documents, test_size=test_size)
#     split_lists = [split(documents) for documents in document_lists]

#     # For some obscure reason this is the main way to flatten arrays in python. Why.
#     train_documents = [train_document for train, test in split_lists for train_document in train]
#     test_documents = [test_document for train, test in split_lists for test_document in test]
#     return (train_documents, test_documents)

# train_subj_docs = subj_docs[:n_train]
# test_subj_docs = subj_docs[n_train:n_instances]
# train_obj_docs = obj_docs[:n_train]
# test_obj_docs = obj_docs[n_train:n_instances]
# training_docs = train_subj_docs+train_obj_docs
# testing_docs = test_subj_docs+test_obj_docs

# sentim_analyzer = SentimentAnalyzer()
# all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

# unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
# sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
# training_set = sentim_analyzer.apply_features(training_docs)
# test_set = sentim_analyzer.apply_features(testing_docs)

# classifier = sentim_analyzer.train(NaiveBayesClassifier.train, training_set)

# for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
#     print('{0}: {1}'.format(key, value))
