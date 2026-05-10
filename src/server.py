import flwr as fl
from client import client_fn
from strategy import PowerOfChoice
from evaluates import gerar_relatorio # Importaremos nossa função de relatório

# Função para agregar a acurácia enviada pelos clientes
def aggregate_metrics(metrics):
    accuracies = [m[1]["accuracy"] for m in metrics]
    lengths = [m[0] for m in metrics]
    weighted_accuracy = sum(a * l for a, l in zip(accuracies, lengths)) / sum(lengths)
    return {"accuracy": weighted_accuracy}

# Metadados do experimento para o relatório
config_experimento = {
    "algoritmo": "Power of Choice",
    "num_clients": 10,
    "num_rounds": 5,
    "d_candidates": 6,
    "k_selected": 3,
    "model_architecture": "Sequential (16 Dense -> 1 Sigmoid)",
    "dataset": "Sintético (10 features, Binário)",
    "optimizer": "Adam"
}

strategy = PowerOfChoice(
    d_candidates=config_experimento["d_candidates"],
    k_selected=config_experimento["k_selected"],
    fraction_evaluate=1.0,
    min_available_clients=config_experimento["num_clients"],
    evaluate_metrics_aggregation_fn=aggregate_metrics, # Adicionado aqui
)

if __name__ == "__main__":
    # Capturamos o objeto 'history' retornado pela simulação
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config_experimento["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config_experimento["num_rounds"]),
        strategy=strategy,
    )

    # Chamamos a função de relatório passando os resultados e a config
    gerar_relatorio(history, config_experimento)
