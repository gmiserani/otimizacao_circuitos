import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use('TkAgg')
import networkx as nx
import matplotlib.pyplot as plt


def convert_crn_to_graph(crn):
    """
    Converte uma CRN para um grafo compatível com o PyTorch Geometric.

    Args:
        crn (dict): Dicionário contendo 'nodes' e 'edges' da CRN.

    Returns:
        Data: Objeto do PyTorch Geometric representando o grafo.
        list: Lista de todos os nós, incluindo os fictícios.
    """
    nodes = crn['nodes']
    edges = crn['edges']

    # Cria mapeamento de nó -> índice
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Adiciona nós fictícios para as reações
    reaction_nodes = []
    graph_edges = []
    for i, (inputs, output) in enumerate(edges):
        if isinstance(inputs, list):
            reaction_node = f"R{i}"
            reaction_idx = len(nodes) + len(reaction_nodes)
            reaction_nodes.append(reaction_node)

            # Adiciona conexões dos reagentes para o nó da reação
            for input_node in inputs:
                graph_edges.append((node_to_idx[input_node], reaction_idx))

            # Adiciona conexão do nó da reação para o produto
            graph_edges.append((reaction_idx, node_to_idx[output]))
        else:
            # Conexão direta para arestas simples (se for do tipo A -> B)
            graph_edges.append((node_to_idx[inputs], node_to_idx[output]))

    # Todos os nós (reais + fictícios)
    all_nodes = nodes + reaction_nodes

    # Criação de tensor de arestas
    edge_index = torch.tensor(graph_edges, dtype=torch.long).t().contiguous()
    node_features = torch.eye(len(all_nodes))  # One-hot encoding

    return Data(x=node_features, edge_index=edge_index), all_nodes



def create_reaction_graph_with_hyperedges():
    """
    Cria um grafo de exemplo representando reações químicas do tipo A + B -> C.
    """
    # Nós (moléculas)
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Reações como hiperarestas
    reactions = [
        (['A', 'B'], 'C'),  # A + B -> C
        (['C', 'D'], 'E'),  # C + D -> E
        (['E', 'F'], 'G')   # E + F -> G
    ]

    # Cria um mapeamento de nome de nó -> índice
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Convertendo hiperarestas para uma representação de grafo bipartido
    edges = []
    reaction_nodes = []
    for i, (inputs, output) in enumerate(reactions):
        reaction_node = f"R{i}"  # Nó fictício para a reação
        reaction_idx = len(node_to_idx) + len(reaction_nodes)  # Novo índice
        reaction_nodes.append(reaction_node)

        # Adiciona conexões dos reagentes para o nó da reação
        for input_node in inputs:
            edges.append((node_to_idx[input_node], reaction_idx))

        # Adiciona conexão do nó da reação para o produto
        edges.append((reaction_idx, node_to_idx[output]))

    # Atualiza os nós totais (moléculas + reações fictícias)
    all_nodes = nodes + reaction_nodes

    # Criação de arestas e características dos nós
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.eye(len(all_nodes))  # One-hot encoding

    return Data(x=node_features, edge_index=edge_index), all_nodes



# Modelo GNN para otimização
class ReactionOptimizerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReactionOptimizerGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)  # Saída final para avaliar cada aresta

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        edge_scores = self.fc(x)  # Predição final para cada aresta
        return edge_scores
def train_model():
    # Cria o grafo de reações químicas
    data, all_nodes = create_reaction_graph_with_hyperedges()

    # Configurações do modelo
    input_dim = data.x.shape[1]  # O número de features do nó corresponde ao número de nós (one-hot encoding)
    hidden_dim = 16
    output_dim = 16
    model = ReactionOptimizerGNN(input_dim, hidden_dim, output_dim)

    # Configuração de otimizador e função de perda
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dados de exemplo para treinamento (rótulos sintéticos: 1 = otimizado, 0 = não otimizado)
    target_labels = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.float).unsqueeze(1)

    # Treinamento
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        predictions = model(data)
        predictions = predictions[:target_labels.size(0)]  # Ajusta o tamanho das predições
        loss = criterion(predictions, target_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Treinamento concluído!")
    return model

def optimize_crn(crn, model):
    """
    Otimiza uma rede de reações químicas (CRN) combinando reações intermediárias em uma única reação.

    Args:
        crn (dict): Um dicionário contendo 'nodes' (lista de moléculas) e 'edges' (lista de reações).
        model (nn.Module): O modelo treinado para otimização.

    Returns:
        dict: CRN otimizada com reações consolidadas.
    """
    # Converte a CRN para o formato de grafo
    data, all_nodes = convert_crn_to_graph(crn)

    # Usa o modelo para prever a importância das arestas
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    # Seleciona as arestas mais importantes (acima de um limiar)
    threshold = 0.5
    selected_edges = [
        crn['edges'][i] for i, score in enumerate(predictions[:len(crn['edges'])])
        if score.item() > threshold
    ]

    # Constrói um grafo com as arestas selecionadas
    reaction_graph = nx.DiGraph()
    for inputs, output in selected_edges:
        for inp in inputs:
            reaction_graph.add_edge(inp, output)

    # Encontra caminhos completos para o produto final
    final_product = 'G'  # Supondo que 'G' seja sempre o produto final
    all_paths = []
    for node in crn['nodes']:
        if reaction_graph.has_node(node):
            try:
                paths = nx.all_simple_paths(reaction_graph, source=node, target=final_product)
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                pass

    # Consolida os caminhos em uma única reação
    if all_paths:
        combined_reaction = ([], final_product)
        for path in all_paths:
            combined_reaction[0].extend(path[:-1])  # Adiciona todos os reagentes do caminho
        combined_reaction = (list(set(combined_reaction[0])), final_product)  # Remove duplicatas

        # Retorna a CRN otimizada com uma única reação consolidada
        return {
            'nodes': crn['nodes'],
            'edges': [combined_reaction]
        }
    else:
        # Se nenhum caminho for encontrado, retorna a CRN original
        return crn



# Função para plotar a CRN
def plot_crn(crn, title):
    G = nx.DiGraph()
    G.add_nodes_from(crn['nodes'])
    G.add_edges_from(crn['edges'])
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=700, font_size=10, font_color='white')
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"Gráfico salvo como {title}.png")


if __name__ == "__main__":
    trained_model = train_model()

    # Exemplo de uma CRN (Chemical Reaction Network)
    crn_example = {
        'nodes': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'edges': [
            (['A', 'B'], 'C'),  # A + B -> C
            (['C', 'D'], 'E'),  # C + D -> E
            (['E', 'F'], 'G')   # E + F -> G
        ]
    }

    # Otimiza a CRN
    optimized_crn = optimize_crn(crn_example, trained_model)

    # Exibe o resultado
    print("CRN Original:")
    print(crn_example)
    print("\nCRN Otimizada:")
    print(optimized_crn)

