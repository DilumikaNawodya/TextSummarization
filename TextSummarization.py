import numpy as np
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx


stop_words = stopwords.words('english')


def read_doc(filepath):
    doc = open(filepath, "r")
    lines = doc.readlines()
    article = lines.split(". ")
    sentences = []
    
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    return sentences


def toLower(sentence):
    new_sentence = []
    for i in sentence:
        new_sentence.append(i.lower())
    return new_sentence
    
def BuildVector(all_words, sentence, stop_words):
    vector = [0] * len(all_words)
    
    for i in sentence:
        if i in stop_words:
            continue
        vector[all_words.index(i)] += 1
    return vector
    
def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []
        
    sent1 = toLower(sent1)
    sent2 = toLower(sent2)
    
    all_words = list(set(sent1+sent2))
    
    vector1 = BuildVector(all_words, sent1, stop_words)
    vector2 = BuildVector(all_words, sent2, stop_words)
    
    return(1 - cosine_distance(vector1, vector2))

def build_similarity_matrix(sentences, stop_words):
    k = len(sentences)
    similarity_matrix = np.zeros((k, k))
    
    for i in range(k):
        for j in  range(k):
            if i==j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
            
    return similarity_matrix

def generate(num):
    
    sentences = read_doc("./Input.txt")

    similarity_matrix = build_similarity_matrix(sentences, stop_words)
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)


    Sorted_Array = []
    for i,j in enumerate(sentences):
        Sorted_Array.append((scores[i],j))
    Sorted_Array.sort()
    Sorted_Array.reverse()

    summary = []
    for i in range(num):
        summary.append(" ".join(Sorted_Array[i][1]))
    print(". ".join(summary))

generate(2)