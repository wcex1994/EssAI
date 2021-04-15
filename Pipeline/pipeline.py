# The specific stylometric calculations in this file are modified versions of code from here.
# We changed them to compose with our existing pipeline infrastructure.
# https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis


#=================================================================
# Load Libraries
#=================================================================

from pipeline.utils import *

import collections as coll
import math
import pickle
import string
import pandas as pd
import numpy as np

import nltk
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pkg_resources
from scipy import stats as s

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
global cmuDictionary
cmuDictionary = None

#=================================================================
# Constants
#=================================================================

SPECIAL_CHARS = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

STOPWORDS = stopwords.words("english") + SPECIAL_CHARS

FUNCTIONAL_WORDS = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """.split()

chall_path = pkg_resources.resource_filename(__name__, "data/dale-chall.pkl")
with open(chall_path, 'rb') as f:
    familiarWords = pickle.load(f)


#=================================================================
# Counts
#=================================================================

def wordCount(text):
    return len(text.split())
               
def uniqueWordCount(text):
    return len(set(text.split()))
               
#=================================================================
# Lexical Richness
#=================================================================
               
def shannonEntropy(text):
    splitText = pd.Series(text.split())
    return s.entropy(splitText.value_counts())


#=================================================================
# Word Length
#=================================================================

def textToWords(text):
    return text.split()

def avgWordLength(words):
    return np.average ([len (word) for word in words if word not in STOPWORDS])

def sentenceLengthByChar(text):
    return len(text)

def specialCharacterCount(text):
    return sum([(char in SPECIAL_CHARS) for char in text])

def countFunctionalWords(words):
    return np.average([(word in FUNCTIONAL_WORDS) for word in words])

#=================================================================
# Syllable Counts
#=================================================================

def syllableCountManual(word):
    word = str(word)
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count

def syllableCount(word):
    d = cmuDictionary
    try:
        syllableCounts = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syllableCounts = syllableCountManual(word)
    return syllableCounts


def avgSyllablesPerWord(words):
    syllables = [syllableCount(word) for word in words]
    return sum(syllables) / max(1, len(words))


#=================================================================
# Word Distributions
#=================================================================


def avgWordFrequencyClass(words):
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])

def hapaxLegemenaTuple(words):

    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return  R, h


def honoreMeasureR(hapaxLegemenaTuple):
    return hapaxLegemenaTuple[0]

def hapaxLegemena(hapaxLegemenaTuple):
    return hapaxLegemenaTuple[1]

def hapaxDisLegemenaTuple(words):
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h

def sichelesMeasureS(hapaxDisLegemenaTuple):
    return hapaxDisLegemenaTuple[0]


def hapaxDisLegemena(hapaxDisLegemenaTuple):
    return hapaxDisLegemenaTuple[1]


def typeTokenRatio(words):
    return len(set(words)) / len(words)


def brunetsMeasureW(words):
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    logged = math.log(N)
    if logged != 0:
        return (V - a) / logged
    else:
        return np.nan
    

def yulesK(words):
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K

def simpsonsIndex(words):
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)

    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    d = (N * (N - 1))
    if d == 1:
        return 1 - (n / d)
    else:
        return np.nan

#=================================================================
# Readability
#=================================================================

def gunningFoxIndex(words, nSentences):
    
    nWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllableCount(word) > 2):
            complexWords += 1
    G = 0.4 * ((nWords / nSentences) + 100 * (complexWords / nWords))
    return G

def fleschReadingEase(words, NoOfsentences = 1):
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllableCount(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return I

def fleschCincadeGradeLevel(words, NoOfSentences):
    scount = 0
    for word in words:
        scount += syllableCount(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F


def daleChallReadabilityFormula(words, nSentences):

    difficult = 0
    adjusted = 0
    nWords = len(words)

    for word in words:
        if word not in familiarWords:
            difficult += 1
    percent = (difficult / nWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (nWords / nSentences) + adjusted
    return D

#=================================================================
# Run Transformation Pipeline
#================================================================= 

def standardPipeline(df, textCol = "Text"):

    transforms = [
        
        (textToWords, "words", textCol),
        (avgWordLength, "avgWordLength", "words"),
        (sentenceLengthByChar, "sentenceLengthByChar", textCol),
        (specialCharacterCount,  "specicalCharacterCount", textCol),
        (avgSyllablesPerWord, "avgSyllablesPerWord", "words"),
        (countFunctionalWords, "countFunctionalWords", "words"),

        (hapaxLegemenaTuple, "hapaxLegemenaTuple", "words"),
        (hapaxLegemena, "hapaxLegemena", "hapaxLegemenaTuple"),
        (honoreMeasureR,  "honoreMeasureR", "hapaxLegemenaTuple"),
        
        (hapaxDisLegemenaTuple, "hapaxDisLegemenaTuple", "words"),
        (hapaxDisLegemena, "hapaxDisLegemena", "hapaxDisLegemenaTuple"),
        (sichelesMeasureS, "sichelesMeasureS", "hapaxDisLegemenaTuple"),
        
        (avgWordFrequencyClass, "avgWordFrequencyClass","words"),
        (typeTokenRatio, "typeTokenRatio","words"),
        (brunetsMeasureW, "brunetsMeasureW","words"),
        (yulesK, "yulesK","words"),
        (simpsonsIndex, "simpsonsIndex","words"),  
        (typeTokenRatio, "typeTokenRatio", "words"),
        
        (wordCount,"wordCount", textCol),
        (uniqueWordCount, "uniqueWordCount", textCol),
        (shannonEntropy, "shannonEntropy", textCol)
    ]
    
    return applyTransforms(df, transforms) 


def readabilityPipeline(groupedDf, sentenceCountCol):
        
    transforms = [
    
      (gunningFoxIndex, "gunningFoxIndex"),
      (daleChallReadabilityFormula, "daleChallReadability"),
      (fleschReadingEase, "fleschReadingEase"),
      (fleschCincadeGradeLevel, "fleschCincadeGradeLevel"),
    ]
    
    for t, col in transforms:
        groupedDf[col] = groupedDf.apply(lambda row : t(row["words"], row[sentenceCountCol]), axis = 1) 
        
    return groupedDf 

def dropIntermediateCols(df):
    COLS =  ["token", "X", 'hapaxDisLegemenaTuple', 'hapaxDisLegemenaTuple', "WordCount"]
    return df.drop(columns = COLS)
