import io
import pickle

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

fname = 'wiki-news-300d-1M.vec'
vectors = load_vectors(fname)
with open("fasttext_vectors.pickle","wb") as outfile:
    pickle.dump(vectors, outfile)
print("Written FastText vectors to pickle file.")