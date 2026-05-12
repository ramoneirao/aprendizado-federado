# Aprendizado Federado

Este projeto implementa um ambiente de simulação para Aprendizado Federado utilizando o framework [Flower (flwr)](https://flower.ai/) e [TensorFlow](https://www.tensorflow.org/).

## Estrutura do Projeto

```text
.
├── src/
│   ├── client.py        # Implementação do cliente Flower e arquitetura do modelo
│   ├── server.py        # Configuração do servidor, métricas e execução da simulação
│   ├── strategy.py      # Estratégias personalizadas (ex: Power of Choice)
│   ├── evaluates.py     # Funções para avaliação e geração de relatórios
│   └── __init__.py      # Inicialização do módulo src
├── main.py              # Script principal de entrada
├── pyproject.toml       # Dependências e configurações do projeto (uv)
├── uv.lock              # Lockfile do uv
└── README.md            # Documentação do projeto
```

## Tecnologias Utilizadas

- **Python 3.11+**
- **Flower (flwr)**: Framework para aprendizado federado.
- **TensorFlow**: Biblioteca para criação e treinamento de modelos de deep learning.
- **uv**: Gerenciador de pacotes e ambientes Python de alta performance.

## Como Executar

Certifique-se de ter o `uv` instalado.

1. Instale as dependências:
   ```bash
   uv sync
   ```

2. Execute a simulação:
   ```bash
   uv run python src/server.py
   ```
