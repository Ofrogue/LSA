from collections import Counter
import nltk
import os
from math import log10
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import codecs


# returns counter with more then number amount of elements
def cutless(c, number):
    k = c.most_common()
    m=dict([token for token in k if token[1]>number])
    k = Counter(m)
    return k

def cutmore(c, number):
    k = c.most_common()
    m=dict([token for token in k if token[1]<number])
    k = Counter(m)
    return k

# [Counter with words, number of words]
def bag_of_words(text):
    text = text.replace('\n',' ')
    ignore_digits=list('''!,'".;:?()%&*-#@+=''')
    ignore_digits = {ord(digit):None for digit in ignore_digits}
    text=text.translate(ignore_digits)
    word_list = text.split()
    # clearing stop-words
    stopwords = nltk.corpus.stopwords.words('russian')
    word_list = [word for word in word_list if word not in stopwords]
    # stemming
    stemmer = nltk.stem.snowball.SnowballStemmer("russian")
    word_list = [stemmer.stem(word) for word in word_list]
    # Counterasing word_list
    c = Counter(word_list)
    c=cutless(c,1)
    return [c,len(word_list)]

# [Counter with words, number of words]
# leave only words which entry in two or more documents
def make_select(c_samples):
    c=Counter()
    for sample in c_samples:
        c += Counter(list(sample[0]))
    c = cutless(c,1)
    dictionary = list(c)
    
    for sample in c_samples:
        sample[0] = Counter(dict([(token, sample[0][token]) for token in dictionary if sample[0][token]>0]))
        
    return dictionary

# get list text files from path 
def get_text_corp(path='text'):
    file_list = [os.path.join(path, file) for file in os.listdir(path)]
    corp=[]
    for file in file_list:
        # corp.append(bag_of_words((codecs.open( file, "r", "utf_8_sig" )).read()))
        corp.append(bag_of_words(open(file,'r').read()))
        print(file)
    return corp

# returns normalized corpus        
def tfidf(corp):
    norm_corp=[]
    for bag in corp:
        items = bag[0].most_common()
        text_len=bag[1]
        doc_amount = len(corp)
        norm_corp.append( [[item[0],item[1]/text_len * log10(doc_amount/entries(corp,item[0]))]
                      for item in items])
    return norm_corp    

# returns how many documents has this word
def entries (corp, word):
    entry = 0
    for bag in corp:
        if bag[0][word]:
            entry += 1
    return entry


# make matrix from normalized corpus with a list of thesaurus
def make_matrix(corpus, thesaurus):
    rows = len(thesaurus)
    colmns = len(corpus)
    matrix = np.zeros((rows,colmns))
    thes = Counter(thesaurus)
    for j, doc in enumerate(corpus):
        words = dict(doc)
        for i, word in enumerate(thesaurus):
            if (words.get(word, 0))!=0:
                matrix[i,j] = words.get(word, 0)
    return matrix

# ortogonilizes and ortonormolizes vectors
def ortonorm(a1,a2):
    b1=a1
    b2=a2 - b1*sum(a2*b1)/sum(b1**2)
    b1 = b1/(sum(b1**2)**0.5)
    b2 = b2/(sum(b2**2)**0.5)
    return b1,b2

# decompose linearly dependent c with b1 b2
def decompose(c, b1, b2):
    a = (c[0]*b2[1] - c[1]*b2[0])
    if a==0:
        alpha = 0
    else:
        alpha = a/(b1[0]*b2[1] - b1[1]*b2[0])
        
    b = (c[0]*b1[1] - c[1]*b1[0])
    if b==0:
        beta = 0
    else:
        beta = b/(b2[0]*b1[1] - b2[1]*b1[0])
    return alpha, beta

# SVD approximation of matrix
def aprox_matr(matr):
    U, W, Vt= np.linalg.svd(matr)
    new_matr = U[:,:2].dot((np.diag(W[:2])).dot(Vt[:2,:]))
    return new_matr

# gets matrix, shows dots on our new plain in ortogonal basis
def plain(matr):
    b1, b2 = ortonorm(matr[0],matr[1])
    vectors  = np.zeros((np.shape(matrix)[0],2))
    for i in range(np.shape(matrix)[0]):
        vectors[i] = decompose(matr[i], b1, b2)
    return vectors

# distance between 2d dots
def distance(a1,a2):
    return (((a1[0] - a2[0])**2)+((a1[1] - a2[1])**2))**0.5

# matrix of distances
def dist_matr(vectors):
    size = np.shape(vectors)[0]
    d_matr = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            dist = distance(vectors[i],vectors[j])
            # value = (1000**(-dist*100))*100
            d_matr[i,j] = dist
    return d_matr

# takes matrix of 2d vectors, returns weight matrix
def weight_connectiont(vectors):
    size = np.shape(vectors)[0]
    w_matr = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            dist = distance(vectors[i],vectors[j])
            value = (1000**(-dist*100))*100
            w_matr[i,j] = value
    return w_matr

# nulls too small values
def cut_matr(matr):
    size= np.shape(matr)
    for i in range(size[0]):
        for j in range(size[1]):
            if matr[i,j]<10:
                matr[i,j] =0
 
def word_clowd(vectors, thesaurus):
    x=[vec[0] for vec in vectors]
    y=[vec[1] for vec in vectors]
    plt.plot(x, y,'sb')
    for i,word in enumerate(thesaurus):
        # plt.annotate(translit(word, 'ru',reversed=True),xy=vectors[i])
        plt.annotate(i ,xy=vectors[i])
    plt.show()
def graph_write(w_matr,thesaurus):
    # ver_head = np.array([['Id', 'Label']])
    ver = pd.DataFrame(list(enumerate(thesaurus)),columns=['Id', 'Label'])
    # verticles = np.concatenate(ver_head, ver)
    # np.savetxt('verticles.csv',ver,delimiter=',')
    print(ver)
    ver.to_csv('verticles.csv',index=False)
    size = np.shape(w_matr)
    edge = np.zeros((size[0]**2,3))
    # edge[0]=['Source', 'Target','Type','Weight']
    counter=0
    
    for i in range(size[0]):
        for j in range(size[1]-i-1):
            if ((w_matr[i,j+i+1]>0) & (i!=j)):
                edge[counter]=[i,j+i+1,w_matr[i,j+i+1]]
                counter+=1
            else: continue
    edge=edge[:counter]
    # np.savetxt('edges.csv', edge, delimiter=',')
    
    edge=pd.DataFrame(edge,columns=['Source', 'Target','Weight'])
    edge.insert(2,value='Undirected',column='Type')
    edge['Source']=edge['Source'].astype('int')
    edge['Target']=edge['Target'].astype('int')
    edge['Weight']=edge['Weight'].astype('int')
    edge.to_csv('edges.csv',index=False)
    print(edge)
    return edge

corpus = get_text_corp()
thesaurus = make_select(corpus)
norm_corp = tfidf(corpus)
matrix = make_matrix(norm_corp, thesaurus)
matrix_2r = aprox_matr(matrix)
vectors = plain(matrix_2r)
w_matr = weight_connectiont(vectors)                    
d_matr= dist_matr(vectors)
cut_matr(w_matr)
graph_write(w_matr,thesaurus)
