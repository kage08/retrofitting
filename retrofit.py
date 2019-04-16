import argparse
import gzip
import math
import numpy
import numpy as np
import re
import sys

from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from itertools import product

from copy import deepcopy

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')  
  for word, values in wordVectors.items():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')      
  outFile.close()
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      newVec = numNeighbours * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord]
      newWordVecs[word] = newVec/(2*numNeighbours)
  return newWordVecs

def get_sim(word1, word2, similarity='path', combine='max'):
  s1 = wn.synsets(word1)
  s2 = wn.synsets(word2)
  if similarity == 'path':
    vals = np.array([x.path_similarity(y) for x,y in product(s1,s2)], dtype=float)
  elif similarity == 'lch':
    vals = np.array([x.lch_similarity(y) for x,y in product(s1,s2)],dtype=float)
  elif similarity == 'wup':
    vals = np.array([x.wup_similarity(y) for x,y in product(s1,s2)],dtype=float)
  elif similarity == 'res':
    from nltk.corpus import reuters
    ic = wn.ic(reuters, False, 0.0)
    vals = np.array([x.res_similarity(y,ic) for x,y in product(s1,s2)],dtype=float)
  elif similarity == 'jcn':
    from nltk.corpus import reuters
    ic = wn.ic(reuters, False, 0.0)
    vals = np.array([x.jcn_similarity(y,ic) for x,y in product(s1,s2)],dtype=float)
  elif similarity == 'lin':
    from nltk.corpus import reuters
    ic = wn.ic(reuters, False, 0.0)
    vals = np.array([x.lin_similarity(y,ic) for x,y in product(s1,s2)],dtype=float)
  
  if combine == 'max':
    return np.nanmax(vals)
  elif combine == 'mean':
    return np.nanmean(vals)
  elif combine == 'min':
    return np.nanmin(vals)
  
  return 0

def retrofit_wnsim(wordVecs, lexicon, numIters, similarity='path',combine='max', alpha=1.0):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      if numNeighbours == 0:
        continue
      weight_total = 0

      newVec = 0.0 * wordVecs[word]

      for ppWord in wordNeighbours:
        wt = get_sim(word,ppWord,similarity,combine)
        newVec += newWordVecs[ppWord] * wt
        weight_total += wt
      newVec += wordVecs[word]*(alpha*weight_total)

      newWordVecs[word] = newVec/((1.0+alpha)*weight_total)
  return newWordVecs


    


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  parser.add_argument("-s", "--similarity", type=str, default='path', help="Similarity")
  parser.add_argument("-c", "--combine", type=str, default='max', help="Combine values")
  args = parser.parse_args()

  wordVecs = read_word_vecs(args.input)
  lexicon = read_lexicon(args.lexicon)
  numIter = int(args.numiter)
  outFileName = args.output
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit_wnsim(wordVecs, lexicon, numIter, similarity=args.similarity, combine = args.combine), outFileName) 
