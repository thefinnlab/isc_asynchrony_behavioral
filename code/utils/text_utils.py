from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re

STOP_WORDS = stopwords.words('english')

def get_pos_tags(text, strip_punc=True):
    """
    Creates POS tags of words in a text corpus. Before tagging,
    performs case and white space normalization and punctuation
    removal before tagging text.
    
    Parameters
    ----------
    text : list of str
        List of text samples (lecture transcript lines, quiz questions,
        or quiz answers) to be processed.

    Returns
    -------
    words_tags : list of tuples
        The word-postag pairings for the list of text samples with 
        preprocessing steps applied to each element.
    """

    # clean spacing, normalize case, strip puncutation
    # (temporarily leave punctuation useful for POS tagging)
    full_text = ' '.join(text) #.lower()
    
    if strip_punc:
        # TLB 10/26/22 - removing ignoring of apostrophe
        # re.sub("[^a-zA-Z\s'-]+", '', full_text)
        full_text = re.sub("[^a-zA-Z\s'-]+", '', full_text)
    # POS tagging (works better on full transcript, more context provided)
    words_tags = pos_tag(full_text.split())

    #case normalize word -> case doesn't matter anymore
    return [(word.lower(), tag) for word, tag in words_tags]

def get_lemma(word, tag, remove_stopwords=True):
    """
    Handles lemmatization of words. Removes stopwords and alpha-numeric
    words from the text.
    
    Parameters
    ----------
    word_tag : tuple of str
        Tuple containing the word to be lemmatized and the accompanying
        WordNet POS.

    Returns
    -------
    lemma : str
        The word-postag pairings for the list of text samples with 
        preprocessing steps applied to each element.
    """
    
    # define some constants only used in this function:
    lemmatizer = WordNetLemmatizer()
    
    # POS tag mapping, format: {Treebank tag (1st letter only): Wordnet}
    tagset_mapping = defaultdict(
        lambda: 'n',   # defaults to noun
        {
            'N': 'n',  # noun types
            'P': 'n',  # pronoun types, predeterminers
            'V': 'v',  # verb types
            'J': 'a',  # adjective types
            'D': 'a',  # determiner
            'R': 'r'   # adverb types
        })
    
    if "'" in word:
        word = word.split("'")[0]
        
    # remove stop words & digits
    if remove_stopwords and word in STOP_WORDS or any(c.isdigit() for c in word):
        return None
    
    # convert Treebank POS tags to WordNet POS tags; lemmatize
    tag = tagset_mapping[tag[0]]
    lemma = lemmatizer.lemmatize(word, tag)
    
    return lemma
        