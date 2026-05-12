from tabulate import tabulate
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd


def exportar_resultados_csv(historicos, configs, filename="resultados_treinamento.csv"):
    lista_dados = []

    for hist, config in zip(historicos, configs):
        losses = hist.losses_distributed
        accuracies = dict(hist.metrics_distributed.get("accuracy", []))
        
        for rodada, loss_val in losses:
            acc_val = accuracies.get(rodada, None)
            
            linha = {
                "Experimento": config.get("nome_experimento", "N/A"),
                "Algoritmo": config["algoritmo"],
                "Rodada": rodada,
                "Loss": loss_val,
                "Acurácia": acc_val,
                "K (Selecionados)": config["k_selected"],
                "D (Candidatos)": config.get("d_candidates", "N/A"),
                "Clientes Totais": config["num_clients"]
            }
            lista_dados.append(linha)

    df = pd.DataFrame(lista_dados)
    
    # Exporta para CSV
    df.to_csv(filename, index=False, encoding='utf-8')

    print(f"\nArquivo CSV '{filename}' gerado com sucesso!")


def gerar_relatorio(history, config):
    print("\n" + "="*50)
    print(f" RESUMO DO CENÁRIO: {config.get('nome_experimento', 'Padrão')}")
    print("="*50)
    
    cenario_data = [
        ["Algoritmo", config["algoritmo"]],
        ["Modelo ML", config["model_architecture"]],
        ["Total de Clientes", config["num_clients"]],
        ["Rodadas", config["num_rounds"]],
        ["Clientes por rodada (k)", config["k_selected"]],
    ]

    if config["algoritmo"] == "Power of Choice":
        cenario_data.append(["PoC (d candidatos)", config["d_candidates"]])

    print(tabulate(cenario_data, tablefmt="fancy_grid"))

def plotar_comparativo(historicos, configs):
    """Gera um gráfico comparando a Acurácia e a Perda de vários experimentos."""
    plt.figure(figsize=(14, 6))

    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 1)
    for hist, config in zip(historicos, configs):
        rodadas = [x[0] for x in hist.losses_distributed]
        losses = [x[1] for x in hist.losses_distributed]
        plt.plot(rodadas, losses, marker='o', label=config['nome_experimento'])
    
    plt.title('Comparativo de Loss Distribuída')
    plt.xlabel('Rodadas de Comunicação')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Gráfico de Acurácia
    plt.subplot(1, 2, 2)
    for hist, config in zip(historicos, configs):
        if "accuracy" in hist.metrics_distributed:
            rodadas = [x[0] for x in hist.metrics_distributed["accuracy"]]
            accs = [x[1] for x in hist.metrics_distributed["accuracy"]]
            plt.plot(rodadas, accs, marker='s', label=config['nome_experimento'])
            
    plt.title('Comparativo de Acurácia Agregada')
    plt.xlabel('Rodadas de Comunicação')
    plt.ylabel('Acurácia')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparativo_poc.png')
    print("\nGráfico comparativo salvo como 'comparativo_poc.png'!")
    plt.show()
