import sys
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, concatenate, add, average
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from score import report_score

#from keras.utils import plot_model 
#from IPython.display import Image

#from google.colab import drive

#drive.mount('/content/drive')
#ROOT_PATH = 'drive/My Drive/'
ROOT_PATH = 'data/'

CLASSIFICATION = ['unrelated', 'discuss', 'agree', 'disagree']
MAX_HEADLINE_LEN = 15
MAX_ARTICLE_LEN = 40
EMBEDDING_DIM = 300
LSTM_DIM = 300
BATCH_SIZE = 128
EPOCH_OPTIONS = [30, 12, 14]

def processing_data(file_name):
  data = pd.read_csv(ROOT_PATH + file_name)
  if 'data' in file_name:
    data = data.sample(frac=1, random_state=10)
    data.reset_index(inplace=True, drop=True)
  headlines = data['Headline']
  articles = data['Article']
  stances = [CLASSIFICATION.index(stance) for stance in data['Stance']]

  headline_word_seq = [text_to_word_sequence(headline, filters="", lower=False, split=' ') for headline in headlines]
  print('90th Percentile HEADLINE Length:', np.percentile([len(seq) for seq in headline_word_seq], 90))
  article_word_seq = [text_to_word_sequence(article, filters="", lower=False, split=' ') for article in articles]
  print('90th Percentile ARTICLE Length:', np.percentile([len(seq) for seq in article_word_seq], 90))
  headlines = [' '.join(headline_seq[:MAX_HEADLINE_LEN]) for headline_seq in headline_word_seq]
  articles = [' '.join(article_seq[:MAX_ARTICLE_LEN]) for article_seq in article_word_seq]

  return headlines, articles, stances


def fitting_tokenizer(data, test, category):
  tokenizer = Tokenizer(filters="", lower=False, split=' ')
  tokenizer.fit_on_texts(data + test)
  if category == 'headline':
    print("Number of words in HEADLINE vocabulary:", len(tokenizer.word_index))
  else:
    print("Number of words in ARTICLE vocabulary:", len(tokenizer.word_index))
  return tokenizer


def padding(headlines, articles, stances, headline_tokenizer, article_tokenizer):
  X_headline = headline_tokenizer.texts_to_sequences(headlines)
  X_headline = pad_sequences(X_headline, maxlen=MAX_HEADLINE_LEN, padding='post', truncating='post')

  X_article = article_tokenizer.texts_to_sequences(articles)
  X_article = pad_sequences(X_article, maxlen=MAX_ARTICLE_LEN, padding='post', truncating='post')

  y = to_categorical(stances)

  return X_headline, X_article, y


def building_embedding_matrix(tokenizer, embeddings):
  embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, EMBEDDING_DIM))
  count = 0
  for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
      try:
          embeddings_vector = embeddings[word]
      except KeyError:
          embeddings_vector = None
          count += 1
      if embeddings_vector is not None:
          embeddings_matrix[i] = embeddings_vector
  print("Not found in total =", count)
  print("embeddings_matrix shape:", embeddings_matrix.shape)
  return embeddings_matrix


def main():
  MODEL_NUM = 3
  if len(sys.argv) > 1:
    MODEL_NUM = int(sys.argv[1])
  
  print("[INFO] Processing training and test data")
  headlines, articles, stances = processing_data("processed_data.csv")
  headlines_test, articles_test, stances_test = processing_data("processed_test.csv")

  print("[INFO] Fitting data into tokenizer")
  headline_tokenizer = fitting_tokenizer(headlines, headlines_test, 'headline')
  article_tokenizer = fitting_tokenizer(articles, articles_test, 'article')

  print("[INFO] Padding data to achieve max length")
  X_headline, X_article, y = padding(headlines, articles, stances, headline_tokenizer, article_tokenizer)
  X_headline_test, X_article_test, y_test = padding(headlines_test, articles_test, stances_test, headline_tokenizer, article_tokenizer)

  print("[INFO] Spliting out validation data")
  X_headline_train, X_headline_val, X_article_train, X_article_val, y_train, y_val = train_test_split(X_headline, X_article, y, random_state=10, test_size=0.1)

  print("[INFO] Loading GloVe embedding from file")
  embeddings = dict()
  with open(ROOT_PATH + 'glove.6B.300d.txt') as f:
      for line in f:
          tokens = line.strip().split(" ")
          embeddings[tokens[0]] = np.array(tokens[1:], dtype='float32')

  print("[INFO] Building embedding matrix for HEADLINE and ARTICLE")
  headline_embeddings_matrix = building_embedding_matrix(headline_tokenizer, embeddings)
  article_embeddings_matrix = building_embedding_matrix(article_tokenizer, embeddings)
  del embeddings

  print("[INFO] Building neural network model")
  headline_input = Input(shape=(MAX_HEADLINE_LEN,), name='headline_input')
  headline_embedding = Embedding(output_dim=EMBEDDING_DIM, 
                input_dim=len(headline_tokenizer.word_index)+1, 
                input_length=MAX_HEADLINE_LEN,
                weights=[headline_embeddings_matrix],
                trainable=False,
                name='headline_embedding')

  article_input = Input(shape=(MAX_ARTICLE_LEN,), name='article_input')
  article_embedding = Embedding(output_dim=EMBEDDING_DIM, 
                input_dim=len(article_tokenizer.word_index)+1, 
                input_length=MAX_ARTICLE_LEN,
                weights=[article_embeddings_matrix],
                trainable=False,
                name='article_embedding')

  if MODEL_NUM == 1:
    # ======= Model 1 ======
    headline_lstm = LSTM(LSTM_DIM, name='headline_lstm')(headline_embedding(headline_input))
    article_lstm = LSTM(LSTM_DIM, name='article_lstm')(article_embedding(article_input))
    merged_vector = average([headline_lstm, article_lstm], name='merged_vector')
    output = Dense(4, activation='softmax', name='output_layer')(merged_vector)
    N_EPOCHS = EPOCH_OPTIONS[0]

  elif MODEL_NUM == 2:
    # ======= Model 2 ======
    headline_lstm = LSTM(LSTM_DIM, return_state=True, name='headline_lstm')
    headline_outputs, state_h, state_c = headline_lstm(headline_embedding(headline_input))
    headline_states = [state_h, state_c]

    article_lstm = LSTM(LSTM_DIM, return_sequences=False, return_state=False, name='article_lstm')
    article_lstm = article_lstm(article_embedding(article_input), initial_state=headline_states)
    dense = Dense(128, activation='relu')(article_lstm)

    output = Dense(4, activation='softmax', name='output_layer')(dense)
    N_EPOCHS = EPOCH_OPTIONS[1]

  else:
    # ======= Model 3 ======
    headline_lstm = Bidirectional(LSTM(LSTM_DIM, return_sequences=False, return_state=True, name='headline_lstm'), merge_mode='concat')
    headline_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = headline_lstm(headline_embedding(headline_input))
    headline_states = [state_h_forward, state_c_forward, state_h_backward, state_c_backward]

    article_lstm = Bidirectional(LSTM(LSTM_DIM, return_sequences=False, return_state=False, name='article_lstm'), merge_mode='concat')
    article_lstm = article_lstm(article_embedding(article_input), initial_state=headline_states)

    output = Dense(4, activation='softmax', name='output_layer')(article_lstm)
    N_EPOCHS = EPOCH_OPTIONS[2]

  print("[INFO] Compiling neural network model")
  model = Model(inputs=[headline_input, article_input], outputs=output)
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  print(model.summary())
  #plot_model(model, to_file=ROOT_PATH + 'model.png', show_layer_names=True, show_shapes=True)
  #Image(ROOT_PATH + 'model.png')

  print("[INFO] Training neural network model")
  model.fit([X_headline_train, X_article_train],
            y_train,
            batch_size=BATCH_SIZE,
            epochs=N_EPOCHS,
            validation_data=([X_headline_val, X_article_val], y_val))

  print("[INFO] Predicting on test data")
  prediction = model.predict([X_headline_test, X_article_test], batch_size=BATCH_SIZE)

  outputs = [np.argmax(pred, axis = -1) for pred in prediction]
  predict_result = [CLASSIFICATION[output] for output in outputs]

  test_data = pd.read_csv(ROOT_PATH + "competition_test_stances.csv")

  real_result = test_data['Stance']
  correct = 0
  for i in range(len(predict_result)):
    if predict_result[i] == real_result[i]:
      correct += 1
  print("[INFO] My accuracy = ", correct / len(predict_result))
  report_score(real_result, predict_result)

  test_data['Stance'] = predict_result
  test_data.to_csv(path_or_buf=ROOT_PATH + "submission.csv", index=False)


if __name__ == '__main__':
  main()