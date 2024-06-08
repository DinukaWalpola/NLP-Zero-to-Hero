from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

sentences = [
        'I love my dog',
        'I love my cat',
        'Do you think my dog is amazing?'
    ]

# by setting the out of vocabulary(oov_token) to something, the tokenizer replaces any unseen data to it
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)  # Create sequences of tokens that represents sentences

###
# After declaring the oov_token now the sentences and the sequences have the same length
# But still it is necessary to have all the sequence to be the same length
# For this we can use ragged tensor or pad sequences
# By using pad sequences all the sequences will have the same length as the longest one, and it assigns 0 to the spaces
# We can use several parameters to the pad_sequence as follows
# padding='post' - Sets the pads to the end of a sequence
# max_len = (int) - If we don't want to set the length of padded sentences to the longest sentences set the desired len
# truncate='post' - chop words from the end if the length is longer or add 'pre' to chop from front
###
padded = pad_sequences(sequences)
print(word_index)
print(sequences)
print(padded)
