from tabulate import tabulate

def gerar_relatorio(history, config):
    print("\n" + "="*50)
    print(" RESUMO DO CENÁRIO SIMULADO")
    print("="*50)
    
    # Apresentação do Cenário
    cenario_data = [
        ["Algoritmo", config["algoritmo"]],
        ["Modelo ML", config["model_architecture"]],
        ["Otimizador", config["optimizer"]],
        ["Dataset", config["dataset"]],
        ["Total de Clientes", config["num_clients"]],
        ["Rodadas de Comunicação", config["num_rounds"]],
        ["Clientes por rodada (k)", config["k_selected"]],
        ["Epochs Locais", config["local_epochs"]],
    ]

    # Adiciona campos do PoC apenas se for a estratégia correta
    if config["algoritmo"] == "Power of Choice":
        cenario_data.append(["PoC (d candidatos)", config["d_candidates"]])

    print(tabulate(cenario_data, tablefmt="fancy_grid"))

    print("\n" + "="*50)
    print(" RESULTADOS POR RODADA")
    print("="*50)

    # Extração de dados do histórico do Flower
    # history.losses_distributed é uma lista de (rodada, valor)
    # history.metrics_distributed contém nossa 'accuracy' agregada
    
    losses = history.losses_distributed
    accuracies = history.metrics_distributed.get("accuracy", [])

    tabela_resultados = []
    for i in range(len(losses)):
        rodada = losses[i][0]
        loss_val = losses[i][1]
        # Verifica se existe acurácia para aquela rodada
        acc_val = accuracies[i][1] if i < len(accuracies) else "N/A"
        
        tabela_resultados.append([
            f"Rodada {rodada}", 
            f"{loss_val:.4f}", 
            f"{acc_val:.2%}" if isinstance(acc_val, float) else acc_val
        ])

    headers = ["Comunicação", "Loss (Distribuída)", "Acurácia"]
    print(tabulate(tabela_resultados, headers=headers, tablefmt="presto"))
    print("="*50 + "\n")