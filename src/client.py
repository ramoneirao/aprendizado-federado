import flwr as fl
import tensorflow as tf
import numpy as np

# 1. Definir uma arquitetura de modelo simples
# def get_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.Input(shape=(10,)),
#         tf.keras.layers.Dense(16, activation="relu"),
#         tf.keras.layers.Dense(1, activation="sigmoid"),
#     ])
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model

# get_model do MNIST:

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),  # 10 dígitos
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# 2. Definir a classe do Cliente Flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Treinamento local
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# 3. Função fábrica que o simulador chamará para criar cada cliente (dados sintéticos)
# def client_fn(cid: str) -> fl.client.Client:
#     np.random.seed(int(cid)) 
    
#     x_train, y_train = np.random.randn(100, 10), np.random.randint(0, 2, 100)
#     x_test, y_test = np.random.randn(20, 10), np.random.randint(0, 2, 20)

#     model = get_model()
#     return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()


# CACHE GLOBAL PARA EVITAR ESTOURO DE MEMÓRIA
_mnist_cache = None

def get_mnist_data():
    global _mnist_cache
    if _mnist_cache is None:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
        x_train_full = x_train_full.reshape(-1, 784) / 255.0
        x_test_full = x_test_full.reshape(-1, 784) / 255.0
        _mnist_cache = (x_train_full, y_train_full, x_test_full, y_test_full)
    return _mnist_cache


# Usando o dataset MNIST
def make_client_fn(num_clients: int):
    def client_fn(cid: str) -> fl.client.Client:
        x_train_full, y_train_full, x_test_full, y_test_full = get_mnist_data()

        train_size = len(x_train_full) // num_clients
        test_size = len(x_test_full) // num_clients
        
        start_train = int(cid) * train_size
        start_test = int(cid) * test_size

        x_train = x_train_full[start_train:start_train + train_size]
        y_train = y_train_full[start_train:start_train + train_size]
        x_test = x_test_full[start_test:start_test + test_size]
        y_test = y_test_full[start_test:start_test + test_size]

        model = get_model()
        return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()
    return client_fn
