from gensim.models import Word2Vec, KeyedVectors
import tensorflow_hub as hub
from scipy.spatial.distance import cosine

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True, limit = 100000)

root = ["cloud computing"]
terms = ["Cloud  computing","cloud Architecture","software computing","centralized computing","cloud (service) computing","computing"]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
vectors = embed(terms)
root_vector = embed(root)

print("root:", len(root_vector[0]))
print("other vector:",len(vectors[0]))

root_vector = root_vector[0]

for i in range (0, len(vectors)):
    similarity = 1 - cosine(root_vector, vectors[i])
    print(terms[i], ": ", similarity)