#!/usr/bin/env python

from gensim import corpora, models, similarities
from itertools import chain
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Infer:
    def __init__(self, path_to_dict=None, path_to_corpus=None, path_to_index=None, path_to_model=None):
        self.path_to_dict = path_to_dict or '/tmp/infer.dict'
        self.path_to_corpus = path_to_corpus or '/tmp/infer.mm'
        self.path_to_index = path_to_index or '/tmp/infer.index'
        self.path_to_model = path_to_model or '/tmp/infer.model'
        self.dictionary = self.corpus = self.lsi_model = self.index = None
        self.existing_corpus = self.documents = None

    def build(self, documents, stopwords=[]):
        self.documents = documents
        texts = [[word for word in document.lower().split() if word not in stopwords] for document in self.documents]
        all_tokens = sum(texts, [])

        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in texts]

        try:
            self.dictionary = corpora.Dictionary.load(self.path_to_dict)
        except IOError:
            self.dictionary = corpora.Dictionary(texts)

        try:
            self.existing_corpus = corpora.MmCorpus(self.path_to_corpus)
        except IOError:
            self.existing_corpus = []

        corpus = [self.dictionary.doc2bow(text, allow_update=True) for text in texts]

        corpora.MmCorpus.serialize(self.path_to_corpus, chain(corpus, self.existing_corpus))
        self.dictionary.save(self.path_to_dict)
        self.corpus = corpora.MmCorpus(self.path_to_corpus)

    def infer(self, query, num_topics=2):
        try:
            self.lsi_model = models.LsiModel.load(self.path_to_model)
            self.lsi_model.add_documents(self.corpus)
        except IOError:
            try:
                self.lsi_model = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=num_topics)
            except ValueError:
                return []

        self.lsi_model.save(self.path_to_model)

        vec_bow = self.dictionary.doc2bow(query.lower().split())
        vec_lsi = self.lsi_model[vec_bow]

        try:
            self.index = similarities.Similarity.load(self.path_to_index)
            self.index.add_documents(self.corpus)
        except IOError:
            self.index = similarities.Similarity(self.path_to_index, self.lsi_model[self.corpus], 500)

        sims = self.index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        return sims

if __name__ == "__main__":
    pass

