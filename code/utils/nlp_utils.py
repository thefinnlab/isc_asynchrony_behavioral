from os.path import join, basename, exists, dirname
from gensim.models import KeyedVectors
from subprocess import run
import numpy as np

def download_model(model_name, data_dir):
    from pathlib import Path
    from os import makedirs
    
    MODELS = {
        'glove.twitter.27B' : ['https://nlp.stanford.edu/data/glove.twitter.27B.zip',
                               'glove.twitter.27B.zip'],
        'glove.42B.300d' : ['https://nlp.stanford.edu/data/glove.42B.300d.zip',
                            'glove.42B.300d.zip'],
        # download directly from google drive doesn't work --> download file manually from link
        'GoogleNews-vectors-negative300': ["https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download",
                                           'GoogleNews-vectors-negative300.bin.gz']
    }
    
    assert (model_name in MODELS)
    
    url, fn = MODELS[model_name]
    
    model_dir = join(data_dir, model_name)
    model_fn = join(model_dir, fn)
    path = Path(fn)
    
    if not exists(model_dir):
        makedirs(model_dir)
    
    # if we don't currently have the model, we have to unzip
    # arg 1 is where to place the downloaded file, arg2 is url
    if not exists(model_fn) and 'drive' not in url: 
        run(f'wget -P {model_dir} -c {url}', shell=True)
    
        if path.suffixes[-1] == '.gz':
            run(f'gzip -d {model_fn}', shell=True)
        elif path.suffixes[-1] == '.zip':
            run(f'unzip {model_fn}', shell=True)
    
    if 'glove' in model_name:
        model_fn = join(dirname(model_fn), basename(model_fn).replace('.zip', '.txt'))

    return model_fn

def load_glove_model(fn):
    '''
    Given the path to a glove model file,  load the model
    into the gensim word2vec format for ease
    '''
    
    d = dirname(fn)
    model_name = basename(fn).replace('.txt', '')
    
    model_fn = join(d, model_name.replace('glove', 'gensim-glove') + '.bin')
    vocab_fn = join(d, model_name.replace('glove', 'gensim-vocab-glove') + '.bin')
    
    # load from preexisting files for ease if exists
    if exists(model_fn):
        print (f'Loading {model_name} from saved .bin file.')
        glove = KeyedVectors.load_word2vec_format(fname=model_fn, 
                                                  fvocab=vocab_fn,
                                                  binary=True)
    else:
        # if the files don't already exist, load the files and save out for next time
        print (f'Loading {model_name} from .txt file and saving into word2vec format.')
        glove = KeyedVectors.load_word2vec_format(fn, binary=False, no_header=True)
        glove.save_word2vec_format(fname=model_fn, 
                                   fvocab=vocab_fn, 
                                   binary=True,
                                   write_header=True)
        
    return glove

def get_basis_vector(model, pos_words, neg_words):
    basis = model[pos_words].mean(0) - model[neg_words].mean(0)
    basis = basis / 1
    return basis

def get_word_score(model, basis, word):
    word_score = np.dot(model[word], basis)
    return word_score

'''Serializable/Pickleable class to replicate the functionality of collections.defaultdict'''
class autovivify_list(dict):
        def __missing__(self, key):
                value = self[key] = []
                return value

        def __add__(self, x):
                '''Override addition for numeric types when self is empty'''
                if not self and isinstance(x, Number):
                        return x
                raise ValueError

        def __sub__(self, x):
                '''Also provide subtraction method'''
                if not self and isinstance(x, Number):
                        return -1 * x
                raise ValueError


def find_word_clusters(labels_array, cluster_labels):
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words

def get_word_clusters(model, cluster, words, norm=True):
    from sklearn.metrics import silhouette_samples
    vectors = np.stack([model.get_vector(word, norm=norm) for word in words])
    
    # Fit model to samples
    cluster.fit(vectors)
    clusters = find_word_clusters(words, cluster.labels_)
    
    scores = silhouette_samples(vectors, cluster.labels_)
    
    return clusters, cluster.labels_, scores