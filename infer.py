#!/usr/bin/env python

from gensim import corpora, models, similarities
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Infer:
    def __init__(self, path_to_dict=None, path_to_corpus=None, path_to_index=None):
        self.path_to_dict = path_to_dict or '/tmp/infer.dict'
        self.path_to_corpus = path_to_corpus or '/tmp/infer.mm'
        self.path_to_index = path_to_index or '/tmp/infer.index'
        self.dictionary = self.corpus = self.documents = None

    def build(self, documents, stopwords=[], update=False):
        self.documents = documents
        texts = [[word for word in document.lower().split() if word not in stopwords] for document in self.documents]
        all_tokens = sum(texts, [])

        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in texts]

        if update:
            try:
                self.dictionary = corpora.Dictionary.load(self.path_to_dict)
            except IOError:
                self.dictionary = corpora.Dictionary(texts)
        else:
            self.dictionary = corpora.Dictionary(texts)

        corpus = [self.dictionary.doc2bow(text, allow_update=True) for text in texts]

        corpora.MmCorpus.serialize(self.path_to_corpus, corpus)
        self.dictionary.save(self.path_to_dict)
        self.corpus = corpora.MmCorpus(self.path_to_corpus)

    def infer(self, query, num_topics=2):
        lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=num_topics)
        vec_bow = self.dictionary.doc2bow(query.lower().split())
        vec_lsi = lsi[vec_bow]

        index = similarities.MatrixSimilarity(lsi[self.corpus])
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        return sims

if __name__ == "__main__":
    pass

