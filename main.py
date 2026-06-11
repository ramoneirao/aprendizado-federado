# pyrefly: ignore [missing-import]
from src.server import run_experiment
# pyrefly: ignore [missing-import]
from src.evaluates import plotar_comparativo, exportar_resultados_csv

def main():
    print("Iniciando bateria de testes do Aprendizado Federado...\n")

    # Configurações base compartilhadas para facilitar
    # base_config = {
    #     "num_clients": 20,
    #     "num_rounds": 15, # Reduzido para teste rápido, aumente se precisar
    #     "local_epochs": 3,
    #     "model_architecture": "CNN MNIST Simples",
    #     "dataset": "MNIST",
    #     "optimizer": "Adam"
    # }


    base_config = {
        "num_clients": 100,
        "num_rounds": 40,
        "local_epochs": 1,
        "model_architecture": "CNN CIFAR-10",
        "dataset": "CIFAR-10",
        "optimizer": "Adam"
    }


    # Definimos os cenários que queremos comparar
    experimentos = [
        {
            **base_config,
            "nome_experimento": "FedAvg (Seleção Aleatória)",
            "algoritmo": "Random Selection",
            "k_selected": 10,
        },
        {
            **base_config,
            "nome_experimento": "PoC Moderado (d=30, k=10)",
            "algoritmo": "Power of Choice",
            "d_candidates": 30,
            "k_selected": 10,
        },
        {
            **base_config,
            "nome_experimento": "PoC Forte (d=60, k=10)",
            "algoritmo": "Power of Choice",
            "d_candidates": 60,
            "k_selected": 10,
        }
    ]

    historicos = []

    # Executa cada configuração e salva o resultado
    for config in experimentos:
        hist = run_experiment(config)
        historicos.append(hist)

    # Plota a comparação ao final
    plotar_comparativo(historicos, experimentos)

    # Exporta os resultados para CSV
    exportar_resultados_csv(historicos, experimentos, "resultados_aprendizado_federado.csv")

if __name__ == "__main__":
    main()
