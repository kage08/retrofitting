from nltk.corpus import wordnet
from queue import Queue
from copy import deepcopy

def get_synonyms_antonyms_hyperhypo(word, syn=True,ant=True, hyper=True,hypo=True):
    synonyms = set()
    antonyms = set()
    hypernyms = set()
    hyponyms = set()
    synsets = wordnet.synsets(word)
    for syn in synsets:
        if syn or ant:
            for l in syn.lemmas():
                synonyms.add(l.name())
                if ant and l.antonyms():
                    antonyms.add(l.antonyms()[0].name())
        if hyper:
            for s in syn.hypernyms():
                for l in s.lemmas():
                    hypernyms.add(l.name())
        if hypo:
            for s in syn.hyponyms():
                for l in s.lemmas():
                    hyponyms.add(l.name())
    
    ans = []
    ans1 = set()
    if syn:
        ans.append(synonyms)
        ans1.update(synonyms)
    if ant:
        ans.append(antonyms)
        ans1.update(antonyms)
    if hyper:
        ans.append(hypernyms)
        ans1.update(hypernyms)
    if hypo:
        ans.append(hyponyms)
        ans1.update(hyponyms)
    
    return ans, ans1

def get_n_hop(word,hops = 1,syn=True,ant=True, hyper=True,hypo=True):
    s = {word}
    words = {word}
    for _ in range(hops):
        s_temp = set()
        while len(s)>0:
            wd = s.pop()
            _, ans = get_synonyms_antonyms_hyperhypo(wd,syn,ant,hyper,hypo)
            for w in ans:
                if w not in words:
                    s_temp.add(w)
        s = deepcopy(s_temp)
        words.update(s_temp)
    
    return words
        