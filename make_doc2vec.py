#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import os
import datetime
from random import shuffle
from contextlib import contextmanager
from timeit import default_timer
#import sys
import re
import gensim
from gensim.models import Doc2Vec
from collections import OrderedDict
from collections import defaultdict
#sys.path.insert(0, '/Users/derek_howard/Desktop/clpsych17')
import config


def get_post_text(row, use_subject):
    try:
        if use_subject:
            subject = row['subject']
            subject = re.sub("^Re: ", "", subject, flags=re.IGNORECASE)
            doc = gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(row['subject']),
                    [str(row['post_id'])])
        else:
            cleaned_body = row['cleaned_body']
            doc = gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(cleaned_body),
                    [str(row['post_id'])])
        return doc

    except TypeError as error:
        print("Error: post_id = ", row['post_id'])
        return None


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def get_features(model):
    """
    Gets doc2vec model features
    Input: Doc2vec model
    Output: pandas dataframe
    """
    tags = pd.DataFrame.from_dict(model.docvecs.doctags, orient='index', dtype=None)
    documentVectors = pd.DataFrame(model.docvecs.doctag_syn0)
    documentVectors = pd.merge(tags, documentVectors, left_on = 'offset', right_index=True)
    documentVectors = documentVectors.drop('doc_count',1) 
    documentVectors = documentVectors.drop('offset',1) 
    return documentVectors 


def add_doc2vec_features(models, use_subject, df):
    corpus = df.apply(get_post_text, axis=1, use_subject=use_subject)
    print('Initial corpus length: ', len(corpus))
    corpus = list(corpus.dropna())
    print("After removing failed parses:", len(corpus))

    df['post_id'] = df['post_id'].astype(str)

    # speed setup by sharing results of 1st model's vocabulary scan
    # PV-DM/concat requires one special NULL word so it serves as template
    models[0].build_vocab(corpus)
    for model in models[0:]:
        model.reset_from(models[0])
        print("Model:" + str(model) + " window:" + str(model.window))

    models_by_name = OrderedDict((str(model) + " window:" + str(model.window), model) for model in models)

    alpha, min_alpha, passes = (0.1, 0.001, 20)
    iterations_per_pass = 10
    alpha_delta = (alpha - min_alpha) / passes

    print("START training {}".format(datetime.datetime.now()))

    for epoch in range(passes):
        shuffle(corpus)  # shuffling gets best results
        for name, train_model in models_by_name.items():
            # Train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha - alpha_delta
            with elapsed_timer() as elapsed:
                # train_model.train(corpus, total_examples=len(corpus), epochs=iterations_per_pass)
                train_model.train(corpus, total_examples=train_model.corpus_count, epochs=iterations_per_pass)
                duration = '%.1f' % elapsed()

        print('Completed pass {} at alpha {}'.format(epoch + 1, alpha))
        alpha -= alpha_delta

    print("END training {}".format(datetime.datetime.now()))

    # rename
    if (use_subject):
        models_by_name['SubjectDoc2VecD100W3'] = models_by_name.pop('Doc2Vec(dbow,d100,n5,mc2,s0.001,t'+str(config.CORES)+') window:3')
    else:
        models_by_name['Doc2VecD50W4'] = models_by_name.pop('Doc2Vec(dbow,d50,n5,mc2,s0.001,t'+str(config.CORES)+') window:4')
        models_by_name['Doc2VecD100W3'] = models_by_name.pop('Doc2Vec(dbow,d100,n5,mc2,s0.001,t'+str(config.CORES)+') window:3')

    # now add it to the dataframe
    for name, train_model in models_by_name.items():
        print("Adding features for " + name)
        features_to_add = get_features(train_model)
        features_to_add.columns = [name+".feature."+str(x) for x in features_to_add.columns]
        df = pd.merge(df, features_to_add, left_on='post_id',right_index=True)
    print("Current shape:" + str(df.shape))
    return df


def main():
    df = pd.read_csv(os.path.join(config.DATA_DIR,
                                  'interim',
                                  'processed_features.csv'))
    print('Number of docs before: {}'.format(df.shape[0]))

    print("Removing Let's count to 1,000,000 posts")
    df = df[df['subject'] != "Let's count to 1,000,000"]
    df = df[df['subject'] != "Re: Let's count to 1,000,000"]
    df = df[df['subject'] != "re: Let's count to 1,000,000"]
    print('Number of docs after removal: {}'.format(df.shape[0]))

    models = [
        # two best found after tests: see test_many_doc2vec_models.ipynb
        # Doc2Vec(dbow,d50,n5,mc2,s0.001,t3) window:4
        Doc2Vec(dm=0, size=50, window=4, min_count=2, workers=config.CORES),
        # Doc2Vec(dbow,d100,n5,mc2,s0.001,t3) window:3
        Doc2Vec(dm=0, size=100, window=3, min_count=2, workers=config.CORES)
    ]

    # model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=2, iter=20,
    #                                      workers=config.CORES)
    #    model.build_vocab(corpus)
    # model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    df = add_doc2vec_features(models, use_subject=False, df=df)
 
    models = [
        # best found after tests: test_many_doc2vec_models.ipynb
        # Doc2Vec(dbow,d100,n5,mc2,s0.001,t3) window:3
        Doc2Vec(dm=0, size=100, window=3, min_count=2, workers=config.CORES)
        ]

    df = add_doc2vec_features(models, use_subject=True, df=df)

    doc2vec_location = os.path.join(config.DATA_DIR, 'interim', 'processed_features_plus_doc2vec.csv')
    print('-- Writing data to ' + doc2vec_location + ' -- ')
    df.to_csv(doc2vec_location, index=False)
    print('-- Done -- ')


if __name__ == "__main__":
    main()
