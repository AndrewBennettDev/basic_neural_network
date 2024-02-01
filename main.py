import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

text = "hello world"

# Create character-level vocabulary
vocab = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

# Convert text to numerical representation
text_as_int = np.array([char_to_idx[char] for char in text])

# Create training data
seq_length = 4
examples_per_epoch = len(text) // (seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Create the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]),
    LSTM(rnn_units,
         return_sequences=True,
         stateful=True,
         recurrent_initializer='glorot_uniform',
         recurrent_activation='sigmoid'),
    Dense(vocab_size)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model
model.fit(dataset, epochs=50)

# Generate text
def generate_text(model, start_string, num_generate=1000):
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx_to_char[predicted_id])

    return start_string + ''.join(text_generated)



# Generate text using the trained model
generated_text = generate_text(model, start_string="hello", num_generate=100)
print(generated_text)
