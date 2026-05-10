import flwr as fl
import tensorflow as tf
import numpy as np

# 1. Definir uma arquitetura de modelo simples
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
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
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# 3. Função fábrica que o simulador chamará para criar cada cliente
def client_fn(cid: str) -> fl.client.Client:
    np.random.seed(int(cid)) 
    
    x_train, y_train = np.random.randn(100, 10), np.random.randint(0, 2, 100)
    x_test, y_test = np.random.randn(20, 10), np.random.randint(0, 2, 20)

    model = get_model()
    return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()
