import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

separator = " ::: "
column_names = ['ID', 'TITLE', 'GENRE', 'SUMMARY']
df = pd.read_csv('data2.txt', sep=separator, names=column_names, engine='python')

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#Memilih 3 kelas
kelas_genre = ['documentary','horror','action']
df = df[df['GENRE'].isin(kelas_genre)]

#Melakukan undersampling agar sampel jumlah sampel data sama untuk masing2 kelas

documentary_data = df[df['GENRE'] == 'documentary']
horror_data = df[df['GENRE'] == 'horror']
action_data = df[df['GENRE'] == 'action']


num_samples_action = len(action_data)

undersampled_documentary= documentary_data.sample(n=num_samples_action, replace=False, random_state=1337)
undersampled_horror = horror_data.sample(n=num_samples_action, replace=False, random_state=1337)

df_undersampled = pd.concat([undersampled_horror, undersampled_documentary, action_data])

df_undersampled = df_undersampled.sample(frac=1, random_state=1337).reset_index(drop=True) #dataset yang digunakan berjumlah 3945 sampel

df_undersampled.drop(columns=['ID', 'TITLE'], inplace=True)

#Menghilangkan kata2 yang dirasa tidak diperlukan dan tanda baca

stop_words = set(stopwords.words('english'))

def remove_stopwords_punctuation(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if (word.lower() not in stop_words and word.lower() not in string.punctuation)]
    return ' '.join(filtered_tokens)

df_undersampled['SUMMARY'] = df_undersampled['SUMMARY'].apply(remove_stopwords_punctuation)

category = pd.get_dummies(df_undersampled.GENRE, dtype=int)
new_df = pd.concat([df_undersampled, category], axis=1)
new_df = new_df.drop(columns='GENRE')

summary = new_df['SUMMARY'].values
y = new_df[['action','documentary','horror']].values

from sklearn.model_selection import train_test_split
summary_latih, summary_test, y_latih, y_test = train_test_split(summary, y, test_size=0.2)

tokenizer = Tokenizer(num_words=10000, oov_token='x')
tokenizer.fit_on_texts(summary_latih) 
 
sekuens_latih = tokenizer.texts_to_sequences(summary_latih)
sekuens_test = tokenizer.texts_to_sequences(summary_test)

padded_latih = pad_sequences(sekuens_latih, maxlen=300, padding='post', truncating='post')
padded_test = pad_sequences(sekuens_test, maxlen=300, padding='post', truncating='post')

import tensorflow as tf

model = Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=15, input_length=300), #Embedding
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24,dropout=0.8, recurrent_dropout=0.8)), #LSTM
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from keras.callbacks import Callback

class StopTrainingCallback(Callback):
    def __init__(self, threshold=0.9):
        super(StopTrainingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= self.threshold:
            print(f"\nAkurasi validasi menyentuh {self.threshold*100:.2f}%.")
            self.model.stop_training = True

stop_callback = StopTrainingCallback(threshold=0.91)

history = model.fit(padded_latih, y_latih, epochs=400,
                    validation_data=(padded_test, y_test),
                    batch_size=32,
                    steps_per_epoch=25,
                    validation_steps=5,
                    callbacks= [stop_callback],
                    verbose=2)



