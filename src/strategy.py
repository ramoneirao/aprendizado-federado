import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, EvaluateRes, Parameters, Scalar
from typing import Dict, List, Optional, Tuple, Union

class PowerOfChoice(fl.server.strategy.FedAvg):
    def __init__(self, d_candidates: int, k_selected: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # d_candidates: conjunto 'A' inicial sorteado aleatoriamente
        # k_selected: subconjunto 'S' escolhido focado nos maiores erros
        self.d_candidates = d_candidates
        self.k_selected = k_selected
        
        # Dicionário de estado interno para mapear Client ID (cid) -> Última Perda (loss)
        self.client_losses: Dict[str, float] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configura a próxima rodada de treinamento aplicando o Power of Choice."""
        
        # Gera as configurações que serão enviadas para os clientes
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)

        available_clients = client_manager.num_available()

        # Rodada 1: Como não temos histórico de perdas, fazemos fallback pro FedAvg padrão
        if server_round == 1 or len(self.client_losses) == 0:
            sample_size = min(self.k_selected, available_clients)
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
            return [(client, fit_ins) for client in clients]

        # --- Lógica do Power of Choice (Rodada 2 em diante) ---
        
        # Passo 1: O servidor amostra um conjunto candidato de tamanho 'd'
        d_size = min(self.d_candidates, available_clients)
        candidate_clients = client_manager.sample(num_clients=d_size)

        # Passo 2: Estima a perda de cada candidato usando o histórico.
        # Se um cliente for novo/nunca avaliado, damos 'inf' para priorizar a escolha dele.
        candidates_with_loss = [
            (client, self.client_losses.get(client.cid, float('inf')))
            for client in candidate_clients
        ]

        # Passo 3: Ordena os candidatos pela MAIOR perda (quem tem mais erro precisa treinar mais)
        candidates_with_loss.sort(key=lambda x: x[1], reverse=True)

        # Passo 4: Seleciona o subconjunto de tamanho 'k'
        selected_clients = [client for client, loss in candidates_with_loss[:self.k_selected]]

        return [(client, fit_ins) for client in selected_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Intercepta os resultados da avaliação para atualizar o mapa de perdas."""
        
        # Atualiza o estado interno com as perdas mais recentes calculadas pelos clientes
        for client, eval_res in results:
            self.client_losses[client.cid] = eval_res.loss

        # Chama o método original do FedAvg para continuar o fluxo normal do Flower
        return super().aggregate_evaluate(server_round, results, failures)