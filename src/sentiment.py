import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import itertools
import random
import math
from typing import Callable, Iterable, Tuple

from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import pandas

import gensim.downloader

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

import numpy

def load_subjective_dataset(n_instances = 5000):
    categories_labels = ["subjective", "objective"]

    subjective_documents = subjectivity.sents(categories="subj")[:n_instances]
    subjective_categories = numpy.array([0]).repeat(n_instances)

    objective_documents = subjectivity.sents(categories="obj")[:n_instances]
    objective_categories = numpy.array([1]).repeat(n_instances)

    documents = subjective_documents + objective_documents
    categories = numpy.concatenate((subjective_categories, objective_categories))

    return train_test_split(documents, categories, stratify=categories)

def clean_tweets(tweets):
    replace_regex = re.compile("#|-+\\s*\\.?\\s*ignore tags\\s*\\.?\\s*-+\\.|\\d+|[?!.]{3,}|@[a-zA-Z0-9_-]+|https?:\/\/[a-zA-Z0-9._/#?-]+|&[a-zA-Z]{2,};")

    return tweets.map(lambda tweet:
        word_tokenize(re.sub(replace_regex, "", str(tweet).lower()))
    )

def load_twitter_sentiment_dataset(nrows = None):
    print("Loading dataset... ", end="", flush=True)
    csv = pandas.read_csv("dataset/twitter_training.csv", header=None, nrows=nrows)
    csv.columns = ["id", "entity", "sentiment", "content"]

    csv = csv.loc[(csv["sentiment"] == "Positive") | (csv["sentiment"] == "Negative")]

    csv = csv[["sentiment", "content"]]

    csv["sentiment"] = csv["sentiment"].map(lambda sentiment: 1 if sentiment == "Positive" else 0)
    csv["content"] = clean_tweets(csv["content"])

    x_train, x_test, y_train, y_test = train_test_split(csv["content"].values, csv["sentiment"].values, stratify=csv["sentiment"].values)

    print("Done!")

    return ([x for x in x_train], [x for x in x_test], [y for y in y_train], [y for y in y_test])

def load_twitter1600k_sentiment_dataset(nrows = None):
    print("Loading dataset... ", end="", flush=True)
    # Note: nrows cannot be passed here because the values are sorted by their positivity
    csv = pandas.read_csv("dataset/models.sentiment00.processed.noemoticon.csv", header=None, encoding="ISO-8859-1")
    csv.columns = ["sentiment","id","date","flag","user","content"]

    negative = csv.loc[csv["sentiment"] == 0]
    positive = csv.loc[csv["sentiment"] == 4]
    if nrows != None:
        negative = negative.head(nrows // 2)
        positive = positive.head(nrows // 2)


    csv = pandas.concat([negative[["sentiment", "content"]], positive[["sentiment", "content"]]], axis=0)

    csv["sentiment"] = csv["sentiment"] // 4
    csv["content"] = clean_tweets(csv["content"])

    x_train, x_test, y_train, y_test = train_test_split(csv["content"].values, csv["sentiment"].values, stratify=csv["sentiment"].values)

    print("Done!")

    return ([x for x in x_train], [x for x in x_test], [y for y in y_train], [y for y in y_test])

glove_vectors = None
def load_glove():
    global glove_vectors
    if glove_vectors == None:
        print("Loading word2vec... ", end="", flush=True)
        glove_vectors = gensim.downloader.load("glove-twitter-25")
        print("Done!")
    else:
        return glove_vectors

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

def train_unigram(n_tokens):
    pipeline = Pipeline([
        ("extract_unigram", UnigramPresence(n_tokens)),
        # ("naive_bayesian", CategoricalNB())
        ("mlp_classifier", MLPClassifier([50, 20, 5], max_iter=1000))
    ])

    print("Loaded")
    pipeline.fit(documents_train, categories_train)

    print(pipeline.score(documents_test, categories_test))
    print(objective_documents[3], categories_labels[pipeline.predict([objective_documents[3]])[0]])

    for unigram in pipeline["extract_unigram"].unigrams:
        proba = pipeline.predict_proba([unigram])[0]
        print(unigram, categories_labels[numpy.argmax(proba)])

    return pipeline


def build_word2vec_model(n_tokens):
    if n_tokens == 7:
        model = models.Sequential()
        model.add(layers.Conv1D(70, 1, padding="same", activation=tf.nn.swish, input_shape=(n_tokens, 25)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv1D(150, 3, padding="same", activation=tf.nn.swish))
        model.add(layers.Dropout(0.33))
        model.add(layers.Conv1D(100, 3, activation=tf.nn.swish))
        model.add(layers.Dropout(0.33))
        # model.add(layers.Conv1D(80, 1, activation=tf.nn.swish))
        # model.add(layers.Dropout(0.4))

        model.add(layers.Flatten(input_shape=(n_tokens - 2, 20)))

        model.add(layers.Dense((n_tokens - 2) * 80, activation=tf.nn.relu))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_tokens * 40 + 20, activation=tf.nn.relu))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_tokens * 10 + 10, activation=tf.nn.relu))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(20, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation=tf.nn.softmax))

        return model
    else:
        model = models.Sequential()
        model.add(layers.Conv1D(100, 3, padding="same", activation=tf.nn.swish, input_shape=(n_tokens, 25)))
        model.add(layers.SpatialDropout1D(0.33))
        model.add(layers.Conv1D(100, 3, padding="same", activation=tf.nn.swish))
        model.add(layers.MaxPool1D(strides=2, padding="same"))
        model.add(layers.SpatialDropout1D(0.33))
        model.add(layers.Conv1D(150, 3, padding="same", activation=tf.nn.swish))
        model.add(layers.SpatialDropout1D(0.33))
        model.add(layers.Conv1D(150, 1, padding="same", activation=tf.nn.swish))

        model.add(layers.Flatten(input_shape=(n_tokens // 2, 150)))

        model.add(layers.Dense(n_tokens * 80, activation=tf.nn.relu))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_tokens * 40 + 20, activation=tf.nn.relu))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_tokens * 10 + 10, activation=tf.nn.relu))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(20, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation=tf.nn.softmax))

        return model

def filter_tokens(document):
    regexp = re.compile("[a-zA-Z]")
    return [word for word in document if (glove_vectors.has_index_for(word.lower()) and bool(re.match(regexp, word)))]

# Returns a modified version of `document`, which retrains all the data, but has matching indices with `filter_tokens`.
def join_tokens(document):
    result = []
    regexp = re.compile("[a-zA-Z]")

    for word in document:
        if glove_vectors.has_index_for(word.lower()) and bool(re.match(regexp, word)):
            result.append(word)
        elif len(result) > 0:
            result[len(result) - 1] += " " + word
        else:
            # TODO: accumulate the first words in a separate list to make up the first token
            print(f"First word in sentence ({word}) is unknown, skipping")

    return result

def get_vectors(document):
    vectors = [glove_vectors.get_vector(word.lower()) for word in document if glove_vectors.has_index_for(word.lower())]

    # Tensorflow does not want to work today it seems
    if len(vectors) == 0:
        return numpy.array([], dtype="float32").reshape((0, 25))

    return numpy.vstack(vectors, dtype="float32").reshape((len(vectors), 25))

def pad_vectors(vectors, n_tokens):
    length = vectors.shape[0]
    if length < n_tokens:
        return numpy.hstack((
            vectors.reshape((length * 25,)),
            numpy.repeat(0, (n_tokens - length) * 25)
        )).reshape((n_tokens, 25))
    else:
        return vectors.reshape((n_tokens, 25))

def augment_vectors(vectors, n_tokens):
    length = vectors.shape[0]
    start = random.randrange(0, max(length - n_tokens, 1))

    return pad_vectors(vectors[start:(start + n_tokens)], n_tokens)

def augment_vectors_gpu(vectors, n_tokens):
    length = tf.shape(vectors)[0]
    start = tf.random.uniform([], 0, length - n_tokens + 1, dtype=tf.dtypes.int32)

    return vectors[start:(start + n_tokens)].to_tensor(shape=[n_tokens, 25])

def evaluate_word2vec_model(model, n_tokens) -> Callable[[str], Iterable[Tuple[str, float]]]:
    def evaluate(text: str) -> Iterable[Tuple[str, float]]:
        tokens = word_tokenize(text)
        vectors = get_vectors(filter_tokens(tokens))
        tokens = join_tokens(tokens)
        assert len(vectors) == len(tokens)

        counts = [0 for _ in range(len(tokens))]
        rates = [0.0 for _ in range(len(tokens))]

        for start in range(max(len(vectors) - n_tokens + 1, 1)):
            inputs = pad_vectors(vectors[start:start + n_tokens], n_tokens)
            prediction = model(inputs.reshape((1, n_tokens, 25)))[0].numpy()
            rate = prediction[1] / (prediction[0] + prediction[1])

            for offset in range(n_tokens):
                if start + offset >= len(vectors):
                    break
                counts[start + offset] += 1
                rates[start + offset] += rate

        for index in range(0, len(vectors)):
            yield (tokens[index], rates[index] / counts[index])

    return evaluate

def train_word2vec(dataset, n_tokens = 5):
    documents_train, documents_test, categories_train, categories_test = dataset

    load_glove()

    # def wrap_as_numpy(callback):
    #     return lambda tensor: callback(tensor.numpy())

    # def wrap_as_numpy2(callback):
    #     return lambda tensor, other: callback(tensor.numpy(), other)

    print("Preparing dataset... ", end="", flush=True)
    document_vectors = [get_vectors(filter_tokens(document)) for document in documents_train]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((
            tf.ragged.stack(document_vectors),
            categories_train
        ))
        .filter(lambda vectors, category: tf.shape(vectors)[0] >= n_tokens)
        .map(lambda vectors, category: (
            augment_vectors_gpu(vectors, n_tokens),
            # tf.py_function(func=wrap_as_numpy2(augment_vectors), inp=[vectors, n_tokens], Tout=tf.float32),
            tf.one_hot(category, 2)
        ))
        .shuffle(128, reshuffle_each_iteration=True)
        .batch(128)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    mapped_documents_test = map(lambda document: augment_vectors(get_vectors(filter_tokens(document)), n_tokens), documents_test)
    mapped_documents_test = numpy.stack([x for x in mapped_documents_test])

    print("Done!")

    model = build_word2vec_model(n_tokens)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    mapped_categories_test = numpy.vstack([tf.one_hot(cat, 2) for cat in categories_test])

    model.fit(
        train_dataset,
        epochs=150,
        validation_data=(mapped_documents_test, mapped_categories_test)
    )

    model.save_weights(f"training/sentiment_{n_tokens}.ckpt")

    loss, accuracy = model.evaluate(mapped_documents_test, mapped_categories_test)
    print(f"Loss: {loss}, accuracy: {accuracy}")

    return (evaluate_word2vec_model(model, n_tokens), model)

def load_word2vec_model(n_tokens = 5):
    model = build_word2vec_model(n_tokens)
    model.load_weights(f"models/sentiment_{n_tokens}.ckpt").expect_partial()

    load_glove()

    return (evaluate_word2vec_model(model, n_tokens), model)

def pretty_print_word2vec(generator, text):
    red = numpy.array([1.0, 0.1, 0.2])
    green = numpy.array([0.1, 1.0, 0.2])
    gray = numpy.array([0.3, 0.3, 0.3])
    spaceless_tokens = ["'ve", "n't", "'m", "'re", "'s", ".", ",", "!", "?"]

    def blend(left, right, amount):
        return left * (1.0 - amount) + right * amount

    first = True
    for token, value in generator(text):
        if value > 0.5:
            color = blend(gray, green, (value - 0.5) * 2.0)
        else:
            color = blend(red, gray, value * 2.0)
        color = [math.floor(color[0] * 255), math.floor(color[1] * 255), math.floor(color[2] * 255)]
        color = f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m"

        if first or (token in spaceless_tokens):
            print(color + token + "\x1b[0m", end="")
        else:
            print(" " + color + token + "\x1b[0m", end="")
        first = False
    print()

if __name__ == "__main__":
    # evaluate, model = train_word2vec(load_twitter1600k_sentiment_dataset(500000), n_tokens=6)
    # evaluate, model = train_word2vec(load_twitter_sentiment_dataset(), n_tokens=6)
    # evaluate, model = train_word2vec(load_subjective_dataset(100))
    evaluate, model = load_word2vec_model(n_tokens=6)

    try:
        text = input("> ")
        while len(text):
            pretty_print_word2vec(evaluate, text)
            text = input("> ")
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass
