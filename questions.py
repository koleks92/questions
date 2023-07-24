import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    
    loaded_files = {}

    for file in files:
        path = os.path.join(directory, file)
        with open(path, 'r', encoding='utf-8') as text:
            content = text.read()
            loaded_files[file] = content

    return loaded_files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []

    p = string.punctuation                      # Punctuaction
    s = nltk.corpus.stopwords.words("english")  # Stopwords


    tokenized = nltk.tokenize.word_tokenize(document)
    for word in tokenized:
        # Remove punctuation and stopwords
        if word not in s and word not in p:
            # Add lowercase
            words.append(word.lower())

    return words



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Get number of documents
    num_of_documents = len(documents)

    # Get set of all words
    all_words = []
    for d in documents:
        for word in documents[d]:
            all_words.append(word)
    all_words = set(all_words)

    idf = {}

    for word in all_words:
        # Number of documents that cointins the word/term
        doc_cont = 0                    
        for d in documents:
            if word in documents[d]:
                doc_cont += 1
        if doc_cont == 0:
            # To avoid zero division (precauction, it should not happen)
            idf[word] = 0
        else:
            idf[word] = math.log(num_of_documents / doc_cont)
    
    return idf
        
    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top_files = {}

    # Interate through each document
    for f in files:
        file_tdidf = 0
        # For each word in a query
        for q in query:
            terms_in_doc = 0
            # For each word in document
            for w in files[f]:
                # If matches the word in query
                if w == q:
                    terms_in_doc += 1
            # Calculate td_idf for a term
            if q in idfs:
                td_idf = terms_in_doc * idfs[q]
            # Add to files total td_idf
            file_tdidf += td_idf
        top_files[f] = file_tdidf

    
    # Sort and get a list of sentences
    files_names = list(s for s, v in sorted(top_files.items(), key=lambda item: item[1], reverse=True))
    
    return files_names[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sen = {}

    # Iterate though each sentence
    for s in sentences:
        s_idfs = 0          # Idf value for a current sentence
        words_in_sen = 0    # How many words in sentence and in query
        # Iterate through each word in query
        for word in sentences[s]:
            # If word in query
            if word in query:
                s_idfs += idfs[word]
                words_in_sen += 1
        if s_idfs != 0:
            sen_length = len(sentences[s])
            qtd = words_in_sen / sen_length # Query Term Denisty
            top_sen[s] = (s_idfs, qtd)

    # Sort and get a list of sentences
    sen_text = list(s for s, v in sorted(top_sen.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True))
    
    return sen_text[0:n]


if __name__ == "__main__":
    main()
