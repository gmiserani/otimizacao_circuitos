import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from gerador import generate_dataset


class ReactionOptimizerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReactionOptimizerGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        edge_scores = self.fc(x)
        return edge_scores


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


def generate_synthetic_labels(crn):
    """
    Gera rótulos sintéticos para uma CRN.
    Marcamos arestas como importantes se forem críticas para alcançar um produto final.

    Args:
        crn (dict): A CRN para a qual os rótulos serão gerados.

    Returns:
        list: Lista de rótulos (0 ou 1) para cada aresta.
    """
    labels = []
    final_products = {'G', 'Z'}
    for inputs, output in crn['edges']:
        if output in final_products or len(inputs) > 2:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def adjust_model_to_graph(model, input_dim):
    """
    Ajusta dinamicamente o modelo para a dimensão de entrada do grafo.

    Args:
        model (ReactionOptimizerGNN): Modelo original.
        input_dim (int): Dimensão de entrada do grafo atual.

    Returns:
        ReactionOptimizerGNN: Modelo ajustado.
    """
    hidden_dim = model.conv1.out_channels
    output_dim = model.conv2.out_channels

    new_model = ReactionOptimizerGNN(input_dim, hidden_dim, output_dim)
    
    with torch.no_grad():
        if model.conv1.in_channels == input_dim:
            new_model.conv1 = model.conv1
        new_model.conv2 = model.conv2
        new_model.fc = model.fc

    return new_model



def train_model(synthetic_crns):
    """
    Treina o modelo com CRNs sintéticas.

    Args:
        synthetic_crns (list): Lista de CRNs sintéticas.

    Returns:
        ReactionOptimizerGNN: Modelo treinado.
    """
    hidden_dim = 16
    output_dim = 16
    model = None
    optimizer = None
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(300):
        epoch_loss = 0

        for crn in synthetic_crns:
            data, all_nodes = convert_crn_to_graph(crn)
            input_dim = data.x.shape[1]

            target_labels = torch.tensor(
                generate_synthetic_labels(crn),
                dtype=torch.float
            ).unsqueeze(1)

            if model is None or model.conv1.in_channels != input_dim:
                model = ReactionOptimizerGNN(input_dim, hidden_dim, output_dim)
                optimizer = optim.Adam(model.parameters(), lr=0.005)

            model.train()
            optimizer.zero_grad()
            predictions = model(data)
            predictions = predictions[:target_labels.size(0)]
            loss = criterion(predictions, target_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(synthetic_crns):.4f}")

    return model



def consolidate_reactions(nodes, edges):
    reaction_graph = nx.DiGraph()
    for inputs, output in edges:
        for inp in inputs:
            reaction_graph.add_edge(inp, output)

    components = [list(c) for c in nx.weakly_connected_components(reaction_graph)]
    optimized_edges = []

    for component in components:
        reactants = [n for n in component if reaction_graph.in_degree(n) > 0]
        product = [n for n in component if reaction_graph.out_degree(n) == 0]
        if product:
            optimized_edges.append((reactants, product[0]))

    return {'nodes': nodes, 'edges': optimized_edges}


def optimize_crn(crn, model_template):
    """
    Otimiza uma CRN baseada em predições do modelo.

    Args:
        crn (dict): CRN contendo 'nodes' e 'edges'.
        model_template (ReactionOptimizerGNN): Modelo treinado.

    Returns:
        dict: CRN otimizada.
    """
    data, _ = convert_crn_to_graph(crn)

    input_dim = data.x.shape[1]
    model = adjust_model_to_graph(model_template, input_dim)

    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(data)).squeeze().tolist()

    threshold = 0.5
    selected_edges = [
        crn['edges'][i] for i, score in enumerate(predictions[:len(crn['edges'])])
        if score > threshold
    ]

    return {
        'nodes': crn['nodes'],
        'edges': selected_edges
    }



if __name__ == "__main__":
    synthetic_crns = generate_dataset(num_samples=10, max_molecules=10, max_reactions=7)

    for i, crn in enumerate(synthetic_crns[:3]):
        print(f"CRN {i + 1}:")
        print(crn)
        print()

    trained_model = train_model(synthetic_crns)

    crn_example = {
    'nodes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z'],
    'edges': [
        (['A', 'B'], 'Z'),
        (['C', 'D'], 'E'),
        (['E', 'F'], 'G')
    ]
}


    optimized_crn = optimize_crn(crn_example, trained_model)
    print("CRN Original:")
    print(crn_example)
    print("\nCRN Otimizada:")
    print(optimized_crn)
