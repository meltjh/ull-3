from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext Cython')

import data
import pickle
from gensim.models import Word2Vec, KeyedVectors
import logging
import numpy as np

#%% Parameters
min_occurence = 2
# Set to False if not preprocessed yet with this minimal occurence.
load_sentences = True
embedding_dim = 300
window_size = 5
num_epochs = 20
start_alpha = 0.025
end_alpha = 0.001
negative_samples = 5

# File names
sent_filename = "sentences_min_{}.pkl".format(str(min_occurence))

model_info = "_{}_{}_{}_{}_{}_{}_{}".format(embedding_dim, window_size, 
                           num_epochs, start_alpha, end_alpha, min_occurence, negative_samples)

model_filename = "skipgram{}".format(model_info)
raw_word_vector_filename = "raw_wordvec{}.txt".format(model_info)
word2id_filename = "word2id{}.txt".format(model_info)

#%% Load or preprocess sentences.

if load_sentences:
    print("Loading sentences")
    with open(sent_filename, 'rb') as f:
        sentences = pickle.load(f)
else:
    sentences = data.read_data('europarl/training', min_occurence)
    print("Saving sentences")
    with open(sent_filename, 'wb') as f:
        pickle.dump(sentences, f)

#%% Create the model and train it.

# Show the progress.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


print("Creating model")
model = Word2Vec(sentences, size = embedding_dim, window = window_size, 
                 min_count = min_occurence, workers = 3, sg = 1, negative = negative_samples,
                 iter = num_epochs)

print("Training the model")
model.train(sentences, total_examples=model.corpus_count, epochs = model.iter,
            start_alpha = start_alpha, end_alpha = end_alpha, compute_loss = True)

# Save the model
print("Saving the model")
model.save(model_filename)

#%% Save word vectors & word2id

print("Saving word vectors & word2id")
data.save_word_vecs_word2id(model, model_info)
