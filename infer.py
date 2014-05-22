#!/usr/bin/env python

''' Wrapper for Gensim LSI Utilities.
    Pranav Ravichandran <me@onloop.net> '''

from gensim import corpora, models, similarities
from itertools import chain
import logging
import os, errno

# Setup basic logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Infer:
    def __init__(self, path=None):
        ''' Load/Initialize dictionary, corpus, index and model from/to disk.
            Use custom path parameter for loading/initializing. '''
        path = path or 'datasets'
        self.__setup_path(path)

        # Paths to individual files.
        self.path_to_dict =  os.path.join(path, 'infer.dict')
        self.path_to_corpus = os.path.join(path, 'infer.mm')
        self.path_to_index =  os.path.join(path, 'infer.index')
        self.path_to_model = os.path.join(path, 'infer.model')

        # Initialize/load from path.
        self.documents = None
        self.dictionary = self.__load_from_fname(lambda: corpora.Dictionary.load(self.path_to_dict))
        self.corpus = self.__load_from_fname(lambda: corpora.MmCorpus(self.path_to_corpus))
        self.lsi_model = self.__load_from_fname(lambda: models.LsiModel.load(self.path_to_model))
        self.index = self.__load_from_fname(lambda: similarities.Similarity.load(self.path_to_index))

    def __load_from_fname(self, read_func):
        ''' Internal helper function for handing exceptions in initialization. '''
        try:
            return read_func()
        except IOError:
            return None

    def __setup_path(self, path):
        ''' Internal helper function to create directory in path if it doesn't exist. '''
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def build(self, documents, stopwords=set([]), update=True, num_topics=2, num_features=500):
        ''' Build the datasets from provided documents and stop words.

            update(default=True): Initialize a new dataset if False. Update existing if True.
            num_topics: Number of topics for the LSI Model.
            num_features: Number of features to be indexed. '''
        self.documents = documents

        # Remove stop words and tokenize. Convert to set for efficiency in lookup.
        stopwords = set(stopwords)
        texts = [[word for word in document.lower().split() if word not in stopwords] for document in self.documents]
        all_tokens = sum(texts, [])

        # Remove words that only appear once.
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in texts]

        ### Load or initialize dictionary based on update arg.
        if update and self.dictionary:
            self.dictionary.add_documents(texts)
        else:
            self.dictionary = corpora.Dictionary(texts)

        self.dictionary.save(self.path_to_dict)
        ######################################################

        ### Load or initialize corpus based on update arg.
        corpus = [self.dictionary.doc2bow(text, allow_update=True) for text in texts]
        corpora.MmCorpus.serialize(self.path_to_corpus, chain(corpus, self.corpus if update and self.corpus else []))

        self.corpus = corpora.MmCorpus(self.path_to_corpus)
        ######################################################

        ### Load or initialize LSI model based on update arg.
        if update and self.lsi_model:
            self.lsi_model.add_documents(self.corpus)
        else:
            self.lsi_model = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=num_topics)

        self.lsi_model.save(self.path_to_model)
        ######################################################

        ### Load or initialize index based on update arg.
        if update and self.index:
            self.index.add_documents(self.corpus)
        else:
            self.index = similarities.Similarity(self.path_to_index, self.lsi_model[self.corpus], num_features)

        self.index.save(self.path_to_index)
        ######################################################

    def infer(self, query):
        ''' Compute similarities between query and documents. '''
        vec_bow = self.dictionary.doc2bow(query.lower().split())
        vec_lsi = self.lsi_model[vec_bow]

        # Perform a similarity query against the corpus.
        sims = self.index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        return sims

if __name__ == "__main__":
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    stoplist = set('for a of the and to in'.split())
    infer = Infer()
    infer.build(documents, stoplist)
    sims = infer.infer("Human computer interaction")
    print sims

