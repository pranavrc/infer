#!/usr/bin/env python

''' Testing Infer on HF Support KB. '''

from infer import Infer

if __name__ == "__main__":
    kb_topics = [topic.rstrip('\n') for topic in open('./kb.txt')]
    stoplist = set('for a of the and to in is are to how do can I ?'.split())
    infer = Infer()
    infer.build(kb_topics, stoplist, update=False, num_topics=len(kb_topics))
    
    sims = infer.infer("How do I subscribe to a ticket")
    print kb_topics[sims[0][0]]
