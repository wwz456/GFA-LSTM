import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer, Flatten,Dropout,Activation,LSTM,BatchNormalization,Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1  
print(tf.__version__)

# CompressFeatures
class CompressFeatures(Layer):
    def call(self, inputs):
        max_compressed = tf.reduce_max(inputs, axis=1)
        mean_compressed = tf.reduce_mean(inputs, axis=1)
        sum_compressed = tf.reduce_sum(inputs, axis=1)
        return max_compressed, mean_compressed, sum_compressed

    def get_config(self):
        return super().get_config()

# ComputeWeights
class ComputeWeights(Layer):
    def __init__(self, mlp, **kwargs):
        super().__init__(**kwargs)
        self.mlp = mlp

    def call(self, inputs):
        max_features, mean_features, sum_features = inputs
        max_out = self.mlp(max_features)
        mean_out = self.mlp(mean_features)
        sum_out = self.mlp(sum_features)

        combined_out = max_out + mean_out + sum_out
        weights = tf.nn.softmax(combined_out, axis=1)
        return weights

    def get_config(self):
        config = super().get_config().copy()
        config['mlp'] = self.mlp.to_json()
        return config

    @classmethod
    def from_config(cls, config):
        mlp = tf.keras.models.model_from_json(config['mlp'])
        return cls(mlp=mlp)

# ApplyWeightsToInput
class ApplyWeightsToInput(Layer):
    def call(self, inputs):
        input_data, weights = inputs
        expanded_weights = tf.expand_dims(weights, 1)
        expanded_weights = tf.tile(expanded_weights, [1, tf.shape(input_data)[1], 1])
        weighted_input = input_data * expanded_weights
        return weighted_input

    def get_config(self):
        return super().get_config()

# create shared MLP
def create_shared_mlp(input_size, hidden_size, output_size):
    inputs = Input(shape=(input_size,))
    x = Dense(hidden_size, activation='relu')(inputs)
    outputs = Dense(output_size)(x)
    model = Model(inputs, outputs)
    return model
from tensorflow.keras import regularizers
# creat model
def build_model(input_shape):
    T, N = input_shape
    hidden_size = N // 2

    inputs = Input(shape=(T, N))

    compress_features_layer = CompressFeatures()(inputs)
    # select if Stadardize the compress features layer or not
    #...
    shared_mlp = create_shared_mlp(input_size=N, hidden_size=hidden_size, output_size=N)
    compute_weights_layer = ComputeWeights(shared_mlp)(compress_features_layer)
    weighted_input = ApplyWeightsToInput()([inputs, compute_weights_layer])
    flatten1 = Flatten()(weighted_input)
    acti0 = Activation('relu')(flatten1)
    # first LSTM
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    dropout0 = Dropout(rate=0.1)(lstm1)
    bn1 = BatchNormalization()(dropout0)
    lstm2 = LSTM(96, return_sequences=True,kernel_regularizer=l1(0.01))(bn1)#,kernel_regularizer=l1(0.01))
    dropout0_1 = Dropout(rate=0.1)(lstm2)
    bn2 = BatchNormalization()(dropout0_1)
    flatten2 = Flatten()(bn2)
    acti2 = Activation('relu')(flatten2)
    # concatenate the 2 output
    concatenated = Concatenate()([acti0, acti2])


    # acti = Activation('relu')(concatenated)
    dropout1 = Dropout(rate=0.3)(concatenated)
    # dense1 = Dense(128,activation='relu')(dropout1)
    # dropout2=Dropout(rate=0.2)(dense1)
    #
    #
    # dense2 = Dense(64,activation='relu')(dropout2)
    # # acti=Activation('relu')(outputs)
    # dropout3=Dropout(rate=0.2)(dense2)
    dense3=Dense(1,activation='linear',kernel_regularizer=regularizers.L1(0.001))(dropout1)
    model = Model(inputs, dense3)
    model.summary()
    return model

# data load
def load_data(m, n, p, sub):
    if sub == 0:
        data_m = np.load(r'E:\gfa-lstm\c{}_nosub.npy'.format(m))
        data_n = np.load(r'E:\gfa-lstm\c{}_nosub.npy'.format(n))
        X_test = np.load(r'E:\gfa-lstm\c{}_nosub.npy'.format(p))
    else:
        data_m = np.load(r'E:\gfa-lstm\c{}_sub{}.npy'.format(m, sub))
        data_n = np.load(r'E:\gfa-lstm\c{}_sub{}.npy'.format(n, sub))
        X_test = np.load(r'E:\gfa-lstm\c{}_sub{}.npy'.format(p, sub))

    label_m = np.load(r'E:\gfa-lstm\data_y{}.npy'.format(m))
    label_n = np.load(r'E:\gfa-lstm\data_y{}.npy'.format(n))
    y_test = np.load(r'E:\gfa-lstm\data_y{}.npy'.format(p))

    X = np.vstack((data_m, data_n))
    y = np.concatenate((label_m, label_n))
    print("X-dimention：", X.shape)
    print("y-dimention：", y.shape)
    return X, y, X_test, y_test

# normalize data
def normalize_data(X, norm_method='std'):
    X_norm = None
    scaler = None
    if len(X.shape) == 2:
        if norm_method == 'std':
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
        elif norm_method == 'min-max':
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(X)
        else:
            print("error")
    else:
        if norm_method == 'std':
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        elif norm_method == 'min-max':
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        else:
            print("error")
    return X_norm, scaler

# plot loss
def plot_loss(epochs, train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_loss, label='Train Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# result plot
def plot_train_test(true_test, pre_test):
    plt.figure(figsize=(10, 3))  # figsize
    plt.xlabel('Cut Number')
    plt.ylabel('Tool Wear(μm)')
    plt.plot(range(len(true_test)), true_test, label='True value', color='r')
    plt.plot(range(len(true_test)), pre_test, label='Predict value', color='g')
    plt.legend()
    plt.tight_layout()
    MAE = [np.abs(a) for a in true_test - pre_test]
    plt.bar(range(len(MAE)), MAE, align='center', label='absolute error', color='orange')
    plt.ylim(0, 240)
    plt.yticks(np.arange(0, 241, 30))
    plt.legend()
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_metrics(y_true, y_pred):
    # MSE
    mse = mean_squared_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # R2
    r2 = r2_score(y_true, y_pred)

    # return all the metrics
    return {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

# data load
X, y, X_test, y_test = load_data(1, 6, 4, 10)

# data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

# normalize data
X_train_norm, scaler = normalize_data(X_train, norm_method='min-max')
X_val_norm = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
X_test_norm = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
max_y = np.max(y_train)
min_y = np.min(y_train)
y_train = (y_train - min_y) / (max_y - min_y)
y_val = (y_val - min_y) / (max_y - min_y)
y_test = (y_test - min_y) / (max_y - min_y)

# set the shape of input data
input_shape = (10, 144)

# create and compile model
model = build_model(input_shape)
model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mae'])

# train the model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
epochs = 150
batch_size = 32

history = model.fit(X_train_norm, y_train, validation_data=(X_val_norm, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

# load the best model
best_model = load_model('best_model.keras', custom_objects={
    'CompressFeatures': CompressFeatures,
    'ComputeWeights': ComputeWeights,
    'ApplyWeightsToInput': ApplyWeightsToInput
})

# predict and evaluate
predictions = best_model.predict(X_val_norm).reshape(-1) * (max_y - min_y) + min_y
y_pre = best_model.predict(X_test_norm).reshape(-1) * (max_y - min_y) + min_y

# plot the result
plot_train_test(y_test * (max_y - min_y) + min_y, y_pre)

# evaluate
evaluate_val = evaluate_metrics(y_val * (max_y - min_y) + min_y, predictions)
evaluate_test = evaluate_metrics(y_test * (max_y - min_y) + min_y,y_pre)

# plot the loss
plot_loss(epochs, history.history['loss'], history.history['val_loss'])
plt.show()
# print the result
print(evaluate_val)
print(evaluate_test)

# save the result
np.save(r'E:\gfa-lstm\c4_gfa-lstm-retrain5.npy', y_pre)
evaluate_test['epochs'] = epochs
evaluate_test['batchsize'] = batch_size

with open(r'E:\gfa-lstm\gfa-lstm_4-retrain5.txt', 'w') as file:
    for metric, value in evaluate_test.items():
        file.write(f"{metric}: {value:.4f}\n")

print("Metrics have been saved to 'metrics.txt'.")
