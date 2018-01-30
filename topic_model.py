#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:32:03 2017

@author: derek_howard
"""

import pandas as pd
import gensim
from gensim.models import LdaModel
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import sys
#sys.path.insert(0, '/Users/derek_howard/Desktop/clpsych17')
import config


models_location = os.path.join(os.getcwd(), 'models')
if not os.path.exists(models_location):
    os.makedirs(models_location)



def clean_text(text):
    try:
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = text.lower()
        tokens = word_tokenize(text)
        final = [lemmatizer.lemmatize(token) for token in tokens
                 if token not in stops]
        return final
    except TypeError as e:
        pass


def build_dictionary(texts=None):
    """
    Input: texts is a list/series of documents (where each doc is a list of
                                                tokens)
    """
    # dict location should be defined in the config
    dict_location = os.path.join(models_location, 'lda_dict.dict')
    if os.path.exists(dict_location):
        print("-- LDA dict found locally")
        dictionary = gensim.corpora.Dictionary.load(dict_location)

    else:
        dictionary = gensim.corpora.Dictionary(texts)
        # probably useful to filter extremes for dictionary
        # dictionary.filter_extremes(no_below=3, no_above=0.5)
        print("Saving a local copy of dictionary for LDA model")
        dictionary.save(dict_location)

    return dictionary


def build_corpus(texts, dictionary):
    # should define location in config
    corpus_location = os.path.join(models_location, 'LDA_MmCorpus.mm')
    if os.path.exists(corpus_location):
        print('MM corpus found locally.')
        corpus = gensim.corpora.MmCorpus.load(corpus_location)
    else:
        corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus


def get_LDA_model(corpus=None, dictionary=None, n_topics=None):
    # should this include n_topics and passes as variables?
    if not os.path.exists(os.path.join(models_location,
                                       'lda_{}topics.model'.format(n_topics))):
        print('Training LDA model with cleaned corpus')
        model = LdaModel(corpus=corpus,
                         num_topics=n_topics,
                         id2word=dictionary,
                         passes=25)
        model.save(os.path.join(models_location,
                                'lda_{}topics.model'.format(n_topics)))

    else:
        print('Loading previously trained model')
        model = LdaModel.load(os.path.join(models_location,
                                           'lda_8topics.model'))

    return model


def get_topic_features(text, ldamodel, num_topics, dictionary):
    # to be applied on original cleaned body text
    # first processes the text into correct format, then feeds it into LDA
    # model to get topic
    try:
        text = clean_text(text)
        bow = dictionary.doc2bow(text)
        ldatopics = ldamodel[bow]
        full_array = gensim.matutils.sparse2full(ldatopics, num_topics)

    except TypeError as e:
        return None

    return full_array


def add_LDA_features(df, model, num_topics, dictionary):
    labeled_rows = df[(df.label.notnull()) | df.predict_me]
    assert(labeled_rows.shape[0] == 1588)

    lda_feats = labeled_rows.cleaned_body.apply(
            (lambda x: pd.Series(get_topic_features(x,
                                                    model,
                                                    num_topics,
                                                    dictionary))))
    lda_feats.rename(columns=lambda x: 'LDA8_{}'.format(x+1), inplace=True)
    lda_feats.fillna(0, inplace=True)

    final = labeled_rows.loc[:, ['post_id']].merge(lda_feats,
                                                   left_index=True,
                                                   right_index=True)
    return final


def main():
    LDA_feats_loc = os.path.join(config.DATA_DIR,
                                 'interim',
                                 'processed_features_LDA.csv')

    df = pd.read_csv(os.path.join(config.DATA_DIR,
                                  'interim',
                                  'processed_features.csv'))#,
                     # usecols=['post_id',
                     #         'subject',
                     #         'label',
                     #         'cleaned_body',
                     #         'predict_me'])

    print('Number of training docs before: {}'.format(df.shape[0]))
    print("Removing Let's count to 1,000,000 posts")
    training_texts = df[df['subject'] != "Let's count to 1,000,000"]
    training_texts = training_texts[training_texts['subject'] != "Re: Let's count to 1,000,000"]
    training_texts = training_texts[training_texts['subject'] != "re: Let's count to 1,000,000"]
    print('Number of training docs after removal:\
          {}'.format(training_texts.shape[0]))

    # remove any of the test dataset from initial LDA training
    # training_df = df[df.predict_me == False]

    # this should only preprocess texts if it will train LDA model directly
    # after, otherwise just load dictionary/corpus/model from pre-trained
    training_texts = training_texts.cleaned_body.apply(clean_text)
    training_texts = training_texts[training_texts.notnull()]

    dictionary = build_dictionary(training_texts)

    corpus = build_corpus(training_texts, dictionary)

    model = get_LDA_model(corpus=corpus,
                          dictionary=dictionary,
                          n_topics=8)

    lda_features = add_LDA_features(df, model, num_topics=8, dictionary=dictionary)
    df = df.merge(lda_features, left_on='post_id', right_on='post_id')

    print('-- Writing LDA features data to {} -- '.format(LDA_feats_loc))
    df.to_csv(LDA_feats_loc, index=False)


if __name__ == "__main__":
    stops = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    main()
