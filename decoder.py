import tensorflow as tf
from tensorflow.keras import layers

# Attention mechanism (Bahdanau)
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features: CNN output (batch, num_features, embedding_dim)
        # hidden: previous hidden state (batch, hidden_dim)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# Decoder with Attention + LSTM
class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = BahdanauAttention(units)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x)

        
        x = self.fc(output)

        return x, state_h, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


#eg
embedding_dim = 256
units = 512
vocab_size = 5000

decoder = Decoder(embedding_dim, units, vocab_size)

# # Sample input
# sample_features = tf.random.normal((64, 49, 256))  # CNN output
# sample_hidden = decoder.reset_state(64)
# sample_input = tf.random.uniform((64, 1), minval=0, maxval=vocab_size, dtype=tf.int32)

# # Forward pass
# output, hidden, attn = decoder(sample_input, sample_features, sample_hidden)

# print("Output shape:", output.shape)
# print("Attention shape:", attn.shape)
