import re
import random
import tokentranslator.translator.grammar.backward as bw
from functools import reduce
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape
from tensorflow.keras import Model

from tensorflow.keras import layers
import tensorflow_datasets as tfds
import time
'''
   E -> E+T|T
   T->T*F|F
   F->(E)|a
'''

grammar = [('E', 'E+T'), ('E', 'T'),
           ('T', 'T*F'), ('T', 'F'),
           ('F', 'a')]


def train_model(size):
    train_data = get_data2(512, size)
    test_data = get_data2(96, size)
    
    model = make_model(size)

    # loss_object = tf.keras.losses.MeanSquaredError()
    loss_object = tf.keras.losses.MeanSquaredError()
    # from_logits=True means expecting logs instead of probs:
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError()
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError()
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(bath_ops, bath_states0, bath_states1):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(bath_ops, bath_states0, training=True)
            # print("predictions:")
            # print(predictions)
            loss = loss_object(bath_states1, predictions)
            # print("loss:")
            # print(loss)
        # print("model.trainable_variables:")
        # print(model.trainable_variables.shape)
        gradients = tape.gradient(loss, model.trainable_variables)
        # print("gradients:")
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(bath_states1, predictions)
        # train_loss(loss)
        train_accuracy(bath_states1, predictions)

    @tf.function
    def test_step(bath_ops, bath_states0, bath_states1):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(bath_ops, bath_states0, training=False)
        t_loss = loss_object(bath_states1, predictions)

        test_loss(bath_states1, predictions)
        # test_loss(t_loss)
        test_accuracy(bath_states1, predictions)

    EPOCHS = 10

    train_losses = []
    test_losses = []

    start_time = time.time()
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        train_data = get_data2(512, size)
        test_data = get_data2(96, size)

        for bath_ops, bath_states0, bath_states1 in train_data:
            # print("bath_ops:")
            # print(bath_ops)
            
            train_step(bath_ops, bath_states0, bath_states1)
            
        for bath_ops, bath_states0, bath_states1 in test_data:
            test_step(bath_ops, bath_states0, bath_states1)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        
        train_losses.append(train_loss.result().numpy())
        test_losses.append(test_loss.result().numpy())
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
    end_time = time.time()
    print("time:")
    print(end_time - start_time)
    print("train_losses:")
    print(train_losses)
    print("test_losses:")
    print(test_losses)
    return((model, train_data, test_data))


# FOR Taken from:
# REF: https://keras.io/getting_started/intro_to_keras_for_researchers/
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z).
    enc = att.Encoder(latent_dim=21,intermediate_dim=32)

    """

    def __init__(self, original_dim, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.reshape = Reshape((original_dim,))
            
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.dense_proj(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit.
    m = att.get_data2(128, 7)
    ops, states0, states1 = next(m.as_numpy_iterator())
    enc = att.Encoder(latent_dim=21,intermediate_dim=32)
    dec = att.Decoder(7,intermediate_dim=32)
    em, ev, e = enc(states0[0])
    dec(e)
    """

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(original_dim, latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        
        return reconstructed
# END FOR


def train_decoder(size, epochs):

    train_data = get_data2(512, size)
    # test_data = get_data2(96, size)
    
    model = VariationalAutoEncoder(size, intermediate_dim=32, latent_dim=21)

    # loss_object = tf.keras.losses.MeanSquaredError()
    loss_object = tf.keras.losses.MeanSquaredError()
    # from_logits=True means expecting logs instead of probs:
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError()
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError()
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(bath_ops, bath_states0, bath_states1):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(bath_states0, training=True)
            # print("predictions:")
            # print(predictions)
            loss = loss_object(bath_states0, predictions)
            loss += sum(model.losses)  # Add KLD term.
            # print("loss:")
            # print(loss)
        # print("model.trainable_variables:")
        # print(model.trainable_variables.shape)
        gradients = tape.gradient(loss, model.trainable_variables)
        # print("gradients:")
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(bath_states0, predictions)
        # train_loss(loss)
        train_accuracy(bath_states0, predictions)

    @tf.function
    def test_step(bath_ops, bath_states0, bath_states1):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(bath_states0, training=False)
        t_loss = loss_object(bath_states0, predictions)
        t_loss += sum(model.losses)  # Add KLD term.
        test_loss(bath_states0, predictions)
        # test_loss(t_loss)
        test_accuracy(bath_states0, predictions)

    EPOCHS = epochs

    train_losses = []
    test_losses = []

    start_time = time.time()
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        train_data = get_data2(512, size)
        test_data = get_data2(96, size)

        for bath_ops, bath_states0, bath_states1 in train_data:
            # print("bath_ops:")
            # print(bath_ops)
            
            train_step(bath_ops, bath_states0, bath_states1)
            
        for bath_ops, bath_states0, bath_states1 in test_data:
            test_step(bath_ops, bath_states0, bath_states1)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        
        train_losses.append(train_loss.result().numpy())
        test_losses.append(test_loss.result().numpy())
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
    end_time = time.time()
    print("time:")
    print(end_time - start_time)
    print("train_losses:")
    print(train_losses)
    print("test_losses:")
    print(test_losses)
    return((model, train_data, test_data))


def make_model(size):
    # train_data = get_data2(512, size)
    # test_data = get_data2(96, size)

    class OpModel(Model):
        '''for predicting operation'''
        def __init__(self):
            super(OpModel, self).__init__()

            # out dim 32, kernel size (3, 3):
            # self.conv1 = Conv2D(32, 3, activation='relu')
            # self.flatten = Flatten()
            self.d0 = Dense(32, activation='relu')
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(2)

        def call(self, x):
            # x = self.conv1(x)
            x = self.d0(x)
            # x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    class StateModel(Model):
        '''for predicting finel state

        m = att.get_data2(128,7)
        ops, states0, states1 = next(m.as_numpy_iterator())
        model(ops, states0)
        model(ops[0][tf.newaxis,...],states0[0][tf.newaxis,...])
        '''
        def __init__(self):
            super(StateModel, self).__init__()

            # out dim 32, kernel size (3, 3):
            # self.conv1 = Conv2D(32, 3, activation='relu')
            # self.flatten = Flatten()

            self.embedding_layer = layers.Embedding(6, 3)

            # for state:
            self.d00 = Dense(32, activation='relu', input_shape=(size*3,))

            # for op:
            self.d01 = Dense(32, activation='relu', input_shape=(3,))

            self.c = layers.Concatenate()
            self.d1 = Dense(128, activation='relu', input_shape=(64, ))
            self.d2 = Dense(size*3)
            self.f0 = Flatten()
            self.reshape0 = Reshape((size*3,), input_shape=(size, 3))
            self.reshape1 = Reshape((1*3,), input_shape=(1, 3))
            self.reshape2 = Reshape((size, 3), input_shape=(size*3,))

        def call(self, op, state):

            # x = self.conv1(x)
            x_state = self.embedding_layer(state)
            x_op = self.embedding_layer(op)

            x_state = self.reshape0(x_state)
            x_op = self.reshape1(x_op)
            print("x_state:")
            print(x_state)
            print("x_op:")
            print(x_op)

            x_state = self.d00(x_state)
            x_op = self.d01(x_op)
            # x = self.flatten(x)
            # print("x_state:")
            # print(x_state)
            # print("x_op:")
            # print(x_op)
            
            # this work for both single and bath:
            x = self.c([x_op, x_state])
            x = self.d1(x)

            # this will not work with batchs:
            # (only like model(ops[0], states0[0])
            #  where ops, states0, states1 = next(m.as_numpy_iterator()))
            # x = self.d1(tf.keras.backend.flatten(tf.concat([x_op, x_state], 0))[tf.newaxis,...])
            x = self.d2(x)
            # x = self.f0(x)
            # self.finel_reshape(x)
            return self.reshape2(x)

    # Create an instance of the model
    model = StateModel()
    return(model)


tok = tfds.features.text.Tokenizer()
tok.tokenize = lambda s: list(s)
tok.join = lambda ts: "".join(ts)                                     

# "0" used for empty:
encoder = tfds.features.text.TokenTextEncoder(["*", "+", "a", "b", "0"],
                                              tokenizer=tok, decode_token_separator="")
embedding_layer = layers.Embedding(6, 3)
'''
test:
label, states = next(dataset.as_numpy_iterator())
s0 = tf.convert_to_tensor(states[0])
s1 = tf.convert_to_tensor(states[1])
tf.py_function(encode, inp=[label, s0, s0], Tout=[tf.string, tf.int64, tf.int64])
'''


def get_data2(count, size):
    data = get_data1(count, size)
    # labels = map(lambda x: x[0], data)
    # vectors = map(lambda x: x[1], data)

    # ndata = tf.data.Dataset.from_tensor_slices((vectors, labels))

    def gen():
        for entry in data:
            print("entry:")
            print(entry)
            yield(entry)

    dataset = tf.data.Dataset.from_generator(
        gen, (tf.string, tf.string),
        (tf.TensorShape([]), tf.TensorShape([None])))

    # return(dataset)

     
    '''
    test:
    label, states = next(dataset.as_numpy_iterator())
    s0 = tf.convert_to_tensor(states[0])
    s1 = tf.convert_to_tensor(states[1])
    tf.py_function(encode, inp=[label, s0, s0], Tout=[tf.string, tf.int64, tf.int64])
    '''
    def encode(label, state0, state1):
        s0 = state0.numpy()
        s1 = state1.numpy()

        # fix globaly:
        fix_size = size-len(s0)
        fixed_state0 = reduce(lambda acc, x: acc+b"0", range(fix_size), s0[:])
        
        # fix out state similiar to input:
        fix_size = size-len(s1)
        fixed_state1 = reduce(lambda acc, x: acc+b"0", range(fix_size), s1[:])
                                             
        # encode strings to numbers:
        et0 = encoder.encode(fixed_state0)
        et1 = encoder.encode(fixed_state1)
        '''
        fix_size = len(s0)-len(s1)
        fixed_state1 = reduce(lambda acc, x: acc+b"0", range(fix_size), s1[:])
                                             
        # encode strings to numbers:
        et0 = encoder.encode(state0.numpy())
        et1 = encoder.encode(fixed_state1)
        '''
        print("label:")
        print(label.numpy())
        l = encoder.encode(label.numpy())
        # print("l:")
        # print(l)
        # embedding_layer(tf.concat([3, 2, 1]))

        # making same shape (vectorize) both states:
        et0 = embedding_layer(tf.convert_to_tensor(et0))
        et1 = embedding_layer(tf.convert_to_tensor(et1))
        l = embedding_layer(tf.convert_to_tensor(l))
        return(l, et0, et1)

    def encode_map(label, states):
        # print("states")
        # print(states)
        # prepare py_function inp args:
        s0 = tf.convert_to_tensor(states[0])
        s1 = tf.convert_to_tensor(states[1])
        label, et0, et1 = tf.py_function(encode, inp=[label, s0, s1],
                                         Tout=[tf.float32, tf.float32, tf.float32])
        # reset shape:
        # et0.set_shape([None])
        # et1.set_shape([None])
        # print("\net0:")
        # print(et0)
        
        # return(label, tf.concat([et0, et1], 0))
        ### return(label, tf.concat([[et0], [et1]], 0))
        # print("shapes: label, et0, et1:")
        # print((label.shape, et0.shape, et1.shape))

        return(tf.keras.backend.flatten(label)[tf.newaxis, ...],
               tf.keras.backend.flatten(et0)[tf.newaxis, ...],
               tf.keras.backend.flatten(et1)[tf.newaxis, ...])

    encoded_data = dataset.map(encode_map)

    BUFFER_SIZE = 1000
    BATCH_SIZE = 32

    shuffled_dataset = encoded_data.shuffle(BUFFER_SIZE)
    m = shuffled_dataset.batch(BATCH_SIZE)
    return(m)
    # return(reduce(lambda acc, x: acc + get_states1(30), range(count), []))


def get_data1(count, size=30):
    return(reduce(lambda acc, x: acc + get_states1(size), range(count), []))


def get_states1(size):
    '''
    - ``size`` -- max size of sent
    '''
    
    sent_list = bw.gen_sent(bw.grammar, size)
    length = len(sent_list)
    print("length:")
    print(length)
    if length > size:
        # ajusting sent length:
        diff = length-size
        delay = diff + 1 if diff % 2 else diff
        print("delay:")
        print(delay)
        sent_list = sent_list[:-delay]

    sent = "".join(sent_list)
    states = bw.gen_states0(sent)
    return(states)


def get_states(steps, size=5):
    results = []
    for step in range(steps):
        eq = [random.choice(["a", "+", "*"]) for i in range(size)]
        eq_str = "".join(eq)
        eq_str = filter_re(eq_str)
        # again, for removing splitted cases (like "++**"->"+*" -> "+"):
        eq_str = filter_re(eq_str)

        try:
            res = eval(eq_str.replace("a", "2"))
        except:
            continue
        if eq_str[0] not in ["+", "*"] and eq_str[-1] not in ["+", "*"]:
            results.append(eq_str)

    return(results)


def filter_re(s):
    eq_str = s
    eq_str = re.subn("aa+", "a", eq_str)[0]
    eq_str = re.subn("\+\++", "+", eq_str)[0]
    eq_str = re.subn("\*\*+", "*", eq_str)[0]
    eq_str = re.subn("\+\*+", "+", eq_str)[0]
    eq_str = re.subn("\*\++", "*", eq_str)[0]
    return(eq_str)


if __name__ == "__main__":
    size = 30
    # print("get_states1(%d):" % size)
    # print(get_states1(size))
    # print("".join(get_data1(size)))
    
    count = 7
    # print("get_data1(%d):" % count)
    # print(get_data1(count))
    
    print("get_data2(%d):" % count)
    print(next(get_data2(count, size).as_numpy_iterator()))


    
