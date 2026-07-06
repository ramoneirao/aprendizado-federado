import flwr as fl
# pyrefly: ignore [missing-import]
from src.client import make_client_fn
# pyrefly: ignore [missing-import]
from src.strategy import PowerOfChoice, RoundRobin, LeastSelectedFirst, LossProportionalSelection
# pyrefly: ignore [missing-import]
from src.evaluates import gerar_relatorio # Importaremos nossa função de relatório


# Função para agregar a acurácia enviada pelos clientes
def aggregate_metrics(metrics):
    accuracies = [m[1]["accuracy"] for m in metrics]
    lengths = [m[0] for m in metrics]
    weighted_accuracy = sum(a * l for a, l in zip(accuracies, lengths)) / sum(lengths)
    return {"accuracy": weighted_accuracy}


def run_experiment(config_experimento):
    """Executa uma simulação baseada no dicionário de configurações passado."""
    print(f"\nIniciando experimento: {config_experimento['nome_experimento']}...")
    
    if config_experimento["algoritmo"] == "Power of Choice":
        strategy = PowerOfChoice(
            d_candidates=config_experimento["d_candidates"],
            k_selected=config_experimento["k_selected"],
            fraction_evaluate=1.0,
            min_available_clients=config_experimento["num_clients"],
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )
    elif config_experimento["algoritmo"] == "Round Robin":
        strategy = RoundRobin(
            k_selected=config_experimento["k_selected"],
            fraction_evaluate=1.0,
            min_available_clients=config_experimento["num_clients"],
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )
    elif config_experimento["algoritmo"] == "Least Selected First":
        strategy = LeastSelectedFirst(
            k_selected=config_experimento["k_selected"],
            fraction_evaluate=1.0,
            min_available_clients=config_experimento["num_clients"],
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )
    elif config_experimento["algoritmo"] == "Loss Proportional":
        strategy = LossProportionalSelection(
            k_selected=config_experimento["k_selected"],
            fraction_evaluate=1.0,
            min_available_clients=config_experimento["num_clients"],
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )
    elif config_experimento["algoritmo"] == "Random Selection":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=config_experimento["k_selected"] / config_experimento["num_clients"],
            fraction_evaluate=1.0,
            min_available_clients=config_experimento["num_clients"],
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )

    client_fn = make_client_fn(config_experimento["num_clients"], config_experimento.get("dataset", "MNIST"))
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config_experimento["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config_experimento["num_rounds"]),
        strategy=strategy,
    )

    gerar_relatorio(history, config_experimento)
    return history
