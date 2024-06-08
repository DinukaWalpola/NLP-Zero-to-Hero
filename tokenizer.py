import tensorflow as tf
from keras_preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
]

# Create Tokenizer object and initialize number of words to store most frequent 100 words
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index # To view the list of encoded words
print(word_index)