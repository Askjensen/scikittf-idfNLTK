# -*- coding: latin -*-
import numpy
import pandas as pd
import sqlalchemy as sa
from Crypto.Cipher import AES
from sklearn import pipeline
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from mysqldb_func import *
from str_func import *
from knn import *
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

def clean_dataframe(streams_df):
    streams_df = streams_df.loc[:,
                 ['Channel', 'Account', 'Stream Type', 'Stream Name', 'Author Display Name', 'Stream Item Type',
                  'Stream Item Text','Parent Stream Item ID']]
   # streams_df['Stream Name'] = [x.decode('latin-1') for x in streams_df['Stream Name']]
    streams_df = streams_df[streams_df['Stream Type'].str.contains("facebook")]
    streams_df = streams_df[streams_df['Stream Item Type'].str.contains("comment")]
    '''
    [u'?Stream Item ID' u'Published Date' u'Initiative' u'Channel' u'Account'
     u'Stream Type' u'Stream ID' u'Stream Name' u'Author Display Name'
     u'Author Username' u'Author Profile URL' u'Stream Item URL' u'Native URL'
     u'Parent Stream Item ID' u'Parent Stream Item URL' u'Stream Item Type'
     u'Stream Item Text' u'Labels' u'Stream Item Notes'
     u'Stream Item Note Timestamp' u'Stream Item Note Display Name'
     u'Stream Item Note Username' u'Current State'
     u'First Responder Display Name' u'First Responder Username'
     u'First Responded Date' u'Responded From Account' u'Responded within SLA']'''
    secret_key = 'AQUHcaBK7POnz8Xw'
    cipher = AES.new(secret_key, AES.MODE_ECB)
    streams_df['comments'] = [decrypt_text
                              (x, cipher) for x in streams_df['Stream Item Text']]
    return streams_df

def tokenize(text):
    tokens = nltk.word_tokenize(text,language='Danish') #danish not default available
    #tokens = nltk.wordpunct_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    # nltk.stem.snowball.DanishStemmer
    #return stems
    return tokens


def main():
    db = connect_to_database()
    cursor = db.cursor()

    #    limit=""
    limit = " limit 100000"
    print 'loading stream  data'
    sql = """SELECT * FROM streams_export as s""" + str(limit)
    engine = sa.create_engine('mysql+mysqldb://mfdev:admin@drmedieforsk01/spredfast', encoding='latin-1')
    streams_df = pd.read_sql(sa.text(sql), engine)
    streams_df = clean_dataframe(streams_df)
    print "N-comments before removing empty comments: " + str(len(streams_df))
    streams_df = streams_df[streams_df.comments != ""]
    print "N-comments after removing empty comments: " + str(len(streams_df))

    ''' old stuff
    texts = {}
    i=0
    for text in streams_df['comments']:
        lowers = text.lower()
        #no_punctuation = lowers.translate(None, string.punctuation)
        no_whitespace = lowers.strip()
        texts[i] = no_whitespace
        i+=1

    #texts = streams_df['comments'].values
    #text2= filterWordsInDocuments(streams_df['comments'])
    # no stop-words removal for now TODO: implement stop-words removal manually for danish
    #TODO: write own tokenizer
    #define vectorizer
    documents = (
    "The sky is blue",
    "The sun is bright",
    "The sun in the sky is bright",
    "We can see the shining sun, the bright sun"
    )
    documents = texts.values()
    documents = [doc for doc in documents if doc != '']


    data = pd.DataFrame({'text': documents})
    '''
    print 'reindiexing to avoid seasonal/time_dependent effects.'
    streams_df = streams_df.reindex(numpy.random.permutation(streams_df.index))

    print 'trying with Kmeans clustering'
    first = True
    frame = pd.DataFrame()
    for num_clusters in range(2, 10):
        tfidf_vectorizer = CountVectorizer(ngram_range=(1, 2),stop_words=stopwords.words('danish'))

        tfidf_matrix = tfidf_vectorizer.fit_transform(streams_df['comments'].values)  # fit the vectorizer to synopses

        #print(tfidf_matrix.shape)
        if True:
            km = KMeans(n_clusters=num_clusters)

            km.fit(tfidf_matrix)

            clusters = km.labels_.tolist()
            # uncomment the below to save your model
            joblib.dump(km, 'output/doc_cluster_' + str(num_clusters) + '.pkl')

        else:
            # to load model:
            km = joblib.load('output/doc_cluster_' + str(num_clusters) + '.pkl')
            clusters = km.labels_.tolist()

        streams_df['kmeans'] = num_clusters
        streams_df['cluster'] = clusters

        if first:
            frame = pd.DataFrame(streams_df, index=streams_df.index,columns=['kmeans','cluster','comments','Stream Name'])
            first = False
        else:
            newframe = pd.DataFrame(streams_df, index=streams_df.index,columns=['kmeans','cluster','comments','Stream Name'])
            frames = [frame, newframe]

            frame = pd.concat(frames)

    try:
        frame.to_csv(path_or_buf='output/clusters_KMeans.csv', sep=';')#,encoding='latin-1')
    except UnicodeDecodeError,e:
        teste=e.message
        for i in e.object.split('\n'):
            print i
            print i.encode('latin-1')
        test=0
    except UnicodeEncodeError,e:
        teste = e.message
        for i in e.object.split('\n'):
            print i
            print i.encode('latin-1')
        test = 0
    test = 0



'''
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize)
    tfidf_vect = TfidfVectorizer()
    #calculate tfidf matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    #tfidf_matrix2 = tfidf_vect.fit_transform(documents)
    #print tfidf_matrix.shape
    #print tfidf_matrix2.shape
    #print cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    def getCosSim(idoc):
            return cosine_similarity(tfidf_matrix[idoc:idoc+1], tfidf_matrix)

    knn = []
    knndist = []
    for i in range(0,len(documents)):
        print getCosSim(i)
        returns = getNeighbors(getCosSim(i), 10)
        knn.append(returns[0])
        knndist.append(returns[1])

    str2 = 'syntes licens historien'
    response = tfidf_matrix.transform([str2])
    print response
    feature_names = tfidf_matrix.get_feature_names()
    for col in response.nonzero()[1]:
        print feature_names[col], ' - ', response[0, col]
    test=0
    #all_texts = filterWordsInDocuments(streams_df['comments'])
    #as_sentences = [' '.join(itxt)  for itxt in all_texts if itxt]
    #raw_txts = [itxt.strip() for itxt in streams_df['comments'] if itxt]
    '''

if __name__ == '__main__':
    main()