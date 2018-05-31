from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
import pickle
import numpy as np
import sys

# Get the stop words from the nltk toolkit and add the punctuation marks to this.
def get_stopwords(language):
    stop_words = stopwords.words(language)
    stop_words = stop_words + [',', '.', ';', ':', '?', '"', "'", '-', '!', ')', '(']
    stop_words = set(stop_words)
    return stop_words

# Get the filtered data.
def read_data(temp_filename, min_occurence):
    stopwords_english = get_stopwords('english')
    en_filtered_sentences = read_data_single_lang(temp_filename + '.en', min_occurence, stopwords_english)
    return en_filtered_sentences

def read_data_single_lang(filename, min_occurence, stop_words):
    sentences = [] 
    all_words = []

    # Obtain all the sentences and all words.
    print("Reading file")
    with open(filename) as f:
        for line in f:
            sentence_words = line.split()
            sentence_words = [word.lower() for word in sentence_words]
            sentences.append(sentence_words)
            all_words += sentence_words
    
    print("Checking allowed words")
    # Count the words before removing words.
    cnt_words_original = Counter(all_words)
    
    words_allowed = dict()
    # Removing least occuring words.
    for word, occur in cnt_words_original.items():
        if (occur > min_occurence) and (word not in stop_words):
            words_allowed[word] = True

    print("Filtering sentences\n")
    # Replace all words that are not in the allowed words by 'UNK' in the sentences.
    filtered_sentences = []
    num_sent = len(sentences)
    progress = 0
    for sentence in sentences:
        progress += 1
        filtered_sentence = []
        for word in sentence:
            if word in words_allowed:
                filtered_sentence.append(word)
            else:
                filtered_sentence.append("UNK")
        filtered_sentences.append(filtered_sentence)
        
        if (progress%1000) == 0:
            sys.stdout.write('\rObtaining sentences: {}/{} ({}%)'.format(progress, num_sent, round(progress/num_sent*100, 2)))
            sys.stdout.flush()
    
    sys.stdout.write('\rFinished obtaining sentences: {}/{}'.format(progress, num_sent))

    # Show the difference in unique words before and after filtering.
    cnt_words_filtered = Counter([word for sentence in filtered_sentences for word in sentence])
    del cnt_words_filtered['UNK']
    print("Number of unique words before vs. after filtering: {} vs. {}".format(len(cnt_words_original.keys()), len(cnt_words_filtered.keys())))
    
    return filtered_sentences

# Save the word2id mappings and the word vectors.
def save_word_vecs_word2id(model, model_info):
    raw__wv_filename = "raw_wordvec{}.txt".format(model_info)
    clean_wv_filename = "wordvec{}.txt".format(model_info)
    word2id_filename = "word2id{}.txt".format(model_info)
    
    model.wv.save_word2vec_format(raw__wv_filename)
    
    with open(raw__wv_filename, 'r') as raw:
        data = raw.read().splitlines(True)

    with open(clean_wv_filename, 'w') as clean:
        clean.writelines(data[1:])
        
    word2id = dict((w, v.index) for w, v in model.wv.vocab.items())
    with open(word2id_filename, 'wb') as f:
        pickle.dump(word2id, f)
