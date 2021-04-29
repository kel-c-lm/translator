import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
MAX_LENGTH = 20

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def to_pairs(doc):
  lines=doc.strip().split('\n')
  pairs=[line.split('\t') for line in lines]
  pairs = [pair for pair in pairs if filterPair(pair)]
  pairs = [list(reversed(p)) for p in pairs]
  return pairs


# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)



def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# load dataset
dataset = load_clean_sentences('/content/drive/MyDrive/me-oe.pkl') #uncomment this if the vocabl thing doesn't work

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
filename = '/content/drive/MyDrive/oe.tsv'
doc = load_doc(filename)
# split into english-old english pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, '/content/drive/MyDrive/me-oe.pkl')
# spot check uncomment this if vocab experiment works
for i in range(100):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename) 

# reduce dataset size
#n_sentences = 1000
#dataset = raw_dataset[:n_sentences, :]
# random shuffle

dataset = load_clean_sentences('/content/drive/MyDrive/me-oe.pkl')
shuffle(dataset)
pairs_length = len(pairs)
eighty_percent = int(len(pairs)*.8)
twenty_percent = pairs_length-eighty_percent
print('Reccomend ', eighty_percent, 'pairs for training and ', twenty_percent, 'for training.')
# split into train/test

train, test = dataset[:2372], dataset[2372:]
save_clean_data(dataset, '/content/drive/MyDrive/me-oe-both.pkl')
save_clean_data(train, '/content/drive/MyDrive/me-oe-train.pkl')
save_clean_data(test, '/content/drive/MyDrive/me-oe-test.pkl')

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint


# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# load datasets
dataset = load_clean_sentences('/content/drive/MyDrive/me-oe-both.pkl')
train = load_clean_sentences('/content/drive/MyDrive/me-oe-train.pkl')
test = load_clean_sentences('/content/drive/MyDrive/me-oe-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare old english tokenizer
oe_tokenizer = create_tokenizer(dataset[:, 1])
oe_vocab_size = len(oe_tokenizer.word_index) + 1
oe_length = max_length(dataset[:, 1])
print('Old English Vocabulary Size: %d' % oe_vocab_size)
print('Old English Max Length: %d' % (oe_length))

# prepare training data
trainX = encode_sequences(oe_tokenizer, oe_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(oe_tokenizer, oe_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# define model
from tensorflow import keras
from keras.callbacks import LearningRateScheduler
import math

model = define_model(oe_vocab_size, eng_vocab_size, oe_length, eng_length, 256)
model.compile(optimizer='Adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
# fit model

model_filename = '/content/drive/MyDrive/model.h5'
checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

initial_learning_rate = 0.001
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
# Fit the model to the training data
history_step_decay = model.fit(
    trainX, 
    trainY, 
    epochs=50, 
    validation_data=(testX, testY),
    batch_size=64,
    callbacks=[LearningRateScheduler(lr_step_decay, verbose=1), checkpoint],
)

from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load datasets
dataset = load_clean_sentences('/content/drive/MyDrive/me-oe-both.pkl')
train = load_clean_sentences('/content/drive/MyDrive/me-oe-train.pkl')
test = load_clean_sentences('/content/drive/MyDrive/me-oe-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare old english tokenizer
oe_tokenizer = create_tokenizer(dataset[:, 1])
oe_vocab_size = len(oe_tokenizer.word_index) + 1
oe_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(oe_tokenizer, oe_length, train[:, 1])
testX = encode_sequences(oe_tokenizer, oe_length, test[:, 1])

# load model
model = load_model('/content/drive/MyDrive/model.h5')
# test on some training sequences
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)