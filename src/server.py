import flwr as fl
from client import client_fn

# Definir a estratégia de agregação (FedAvg por padrão)
# É aqui que, futuramente, implementaremos a customização para o Power of Choice
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,         # Amostra 50% dos clientes disponíveis para treinamento por rodada
    fraction_evaluate=0.5,    # Amostra 50% dos clientes para avaliação
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=10, # Total de clientes na simulação
)

# Iniciar a simulação
if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3), # Executa por 3 rodadas de comunicação
        strategy=strategy,
    )