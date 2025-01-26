import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use('TkAgg')
import networkx as nx
import matplotlib.pyplot as plt
from gerador import generate_dataset

def convert_crn_to_graph(crn):
    nodes = crn['nodes']
    edges = crn['edges']
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    reaction_nodes = []
    graph_edges = []

    for i, (inputs, output) in enumerate(edges):
        reaction_node = f"R{i}"
        reaction_idx = len(nodes) + len(reaction_nodes)
        reaction_nodes.append(reaction_node)
        for input_node in inputs:
            graph_edges.append((node_to_idx[input_node], reaction_idx))
        graph_edges.append((reaction_idx, node_to_idx[output]))

    all_nodes = nodes + reaction_nodes
    edge_index = torch.tensor(graph_edges, dtype=torch.long).t().contiguous()
    node_features = torch.eye(len(all_nodes))
    return Data(x=node_features, edge_index=edge_index), all_nodes

class ReactionOptimizerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReactionOptimizerGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        edge_scores = self.fc(x)
        return edge_scores


def generate_synthetic_labels(crn):
    labels = []
    for inputs, output in crn['edges']:
        # Arestas com mais reagentes ou conectadas ao produto final têm maior prioridade
        labels.append(1 if len(inputs) > 1 else 0)
    return labels


def train_model(synthetic_crns):
    hidden_dim = 16
    output_dim = 16
    model = None
    optimizer = None
    positive_weight = torch.tensor([2.0])  # Pesa mais as arestas importantes
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)


    for epoch in range(100):
        epoch_loss = 0

        for crn in synthetic_crns:
            # Converte a CRN para grafo
            data, _ = convert_crn_to_graph(crn)
            input_dim = data.x.shape[1]

            # Gera rótulos sintéticos
            target_labels = torch.tensor(
                generate_synthetic_labels(crn),
                dtype=torch.float
            ).unsqueeze(1)

            # Recria o modelo se o número de nós mudar
            if model is None or model.conv1.in_channels != input_dim:
                model = ReactionOptimizerGNN(input_dim, hidden_dim, output_dim)
                optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Treinamento
            model.train()
            optimizer.zero_grad()
            predictions = torch.sigmoid(model(data))
            predictions = predictions[:target_labels.size(0)]
            loss = criterion(predictions, target_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(synthetic_crns):.4f}")

    print("Treinamento concluído!")
    return model

def optimize_crn(crn, model_template):
    """
    Otimiza uma rede de reações químicas (CRN) combinando reações intermediárias em uma única reação.

    Args:
        crn (dict): Um dicionário contendo 'nodes' (lista de moléculas) e 'edges' (lista de reações).
        model_template (nn.Module): Um modelo base treinado.

    Returns:
        dict: CRN otimizada com reações consolidadas.
    """
    # Converte a CRN para o formato de grafo
    data, all_nodes = convert_crn_to_graph(crn)

    # Ajusta o modelo para o número de nós da CRN atual
    input_dim = data.x.shape[1]
    hidden_dim = model_template.conv1.out_channels
    output_dim = model_template.fc.in_features

    # Recria o modelo com dimensões compatíveis
    model = ReactionOptimizerGNN(input_dim, hidden_dim, output_dim)

    # Ajusta os pesos do modelo treinado para corresponder às novas dimensões
    pretrained_state = model_template.state_dict()
    current_state = model.state_dict()

    # Atualiza apenas os pesos compatíveis
    for name, param in pretrained_state.items():
        if name in current_state and param.size() == current_state[name].size():
            current_state[name].copy_(param)

    model.load_state_dict(current_state)

    # Usa o modelo para prever a importância das arestas
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(data))

    # Seleciona as arestas mais importantes (acima de um limiar)
    threshold = 0.3
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


def plot_crn(crn, title):
    G = nx.DiGraph()
    G.add_nodes_from(crn['nodes'])
    G.add_edges_from(crn['edges'])
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=700, font_size=10, font_color='white')
    plt.title(title)
    plt.savefig(f"{title}.png")
    print(f"Gráfico salvo como {title}.png")

# Ajuste no script principal
if __name__ == "__main__":
    # Gera um conjunto de CRNs sintéticas
    synthetic_crns = generate_dataset(num_samples=10, max_molecules=10, max_reactions=7)

    # Exibe exemplos de CRNs geradas
    for i, crn in enumerate(synthetic_crns[:3]):
        print(f"CRN {i + 1}:")
        print(crn)
        print()

    # Treina o modelo usando as CRNs sintéticas
    trained_model = train_model(synthetic_crns)

    # Exemplo de uma CRN para otimização
    crn_example = {
        'nodes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z'],
        'edges': [
            (['A', 'B'], 'Z'),  # A + B -> C
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