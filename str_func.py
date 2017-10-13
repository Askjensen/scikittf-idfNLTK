import base64

import csv

import gensim
import stop_words
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='danish')
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]

def writeCSVfileFromTextDict(outpath,datadict):
    outfile = open(outpath, "w")
    csv_file = csv.writer(outfile, delimiter=';', dialect='excel')
    for i in range(len(datadict)):
        row = [str(i),datadict[i].encode('utf-8'),str(datadict.dfs[i])]
        csv_file.writerow(row)
    outfile.close()

def filterWordsInDocuments(documents):
    stopwords = stop_words.get_stop_words('danish')
    # remove words that appear only once
    texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    # # utf-8 encode the text
    #for itext in range(0, len(texts)):
    #    texts[itext] = [x.encode('utf-8') for x in texts[itext]]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    # if textsshould be printed
    # from pprint import pprint  # pretty-printer
    # pprint(texts)
    texts = [itxt for itxt in texts if itxt]
    return texts


def createDictionary(texts):
    return gensim.corpora.Dictionary(texts)
    #dictionary.save('out/' + str(data.keys()[0]) + '.dict')  # store the dictionary, for future reference
    #return dictionary


def createCorpus(dictionary, texts):
    return [dictionary.doc2bow(text) for text in texts]
    # gensim.corpora.MmCorpus.serialize('out/' + str(data.keys()[0]) + '.mm', corpus)  # store to disk, for later use
    #return corpus


def decrypt_text(text,cipher):
    try:
        return cipher.decrypt(base64.b64decode(text))
    except:
        return ''

