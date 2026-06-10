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

def get_model_cifar():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
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
_cifar_cache = None

def get_mnist_data():
    global _mnist_cache
    if _mnist_cache is None:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
        x_train_full = x_train_full.reshape(-1, 784) / 255.0
        x_test_full = x_test_full.reshape(-1, 784) / 255.0
        _mnist_cache = (x_train_full, y_train_full, x_test_full, y_test_full)
    return _mnist_cache

def get_cifar_data():
    global _cifar_cache
    if _cifar_cache is None:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
        x_train_full = x_train_full.astype("float32") / 255.0
        x_test_full = x_test_full.astype("float32") / 255.0
        y_train_full = y_train_full.flatten()
        y_test_full = y_test_full.flatten()
        _cifar_cache = (x_train_full, y_train_full, x_test_full, y_test_full)
    return _cifar_cache

def _make_noniid_partitions(y, num_clients, num_classes=10, alpha=0.5, seed=42):
    """Partição Não-IID via distribuição de Dirichlet. Alpha menor = mais heterogêneo."""
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = np.split(cls_indices, (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1])
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())
    return client_indices


# Usando o dataset MNIST ou CIFAR-10
def make_client_fn(num_clients: int, dataset: str = "MNIST"):
    # Pré-computa partições Não-IID para CIFAR-10 antes de criar os closures
    if dataset == "CIFAR-10":
        x_train_full, y_train_full, x_test_full, y_test_full = get_cifar_data()
        train_partitions = _make_noniid_partitions(y_train_full, num_clients)
        test_partitions = _make_noniid_partitions(y_test_full, num_clients, seed=99)

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)

        if dataset == "CIFAR-10":
            train_idx = train_partitions[client_id]
            test_idx = test_partitions[client_id]
            x_train = x_train_full[train_idx]
            y_train = y_train_full[train_idx]
            x_test = x_test_full[test_idx]
            y_test = y_test_full[test_idx]
            model = get_model_cifar()
        else:
            x_tr, y_tr, x_te, y_te = get_mnist_data()
            train_size = len(x_tr) // num_clients
            test_size = len(x_te) // num_clients
            x_train = x_tr[client_id * train_size:(client_id + 1) * train_size]
            y_train = y_tr[client_id * train_size:(client_id + 1) * train_size]
            x_test = x_te[client_id * test_size:(client_id + 1) * test_size]
            y_test = y_te[client_id * test_size:(client_id + 1) * test_size]
            model = get_model()

        return FlowerClient(model, x_train, y_train, x_test, y_test).to_client()
    return client_fn
