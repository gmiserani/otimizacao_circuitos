import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Exemplo de definição do grafo de reações químicas
def create_reaction_graph():
    """
    Cria um grafo de exemplo representando reações químicas.
    Nós: moléculas (A, B, C, D, E, F, G).
    Arestas: reações químicas entre elas.
    """
    # Representação de nós (moléculas)
    nodes = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6
    }

    # Arestas do grafo (reações)
    edges = [
        (nodes['A'], nodes['C']),  # A + B -> C
        (nodes['B'], nodes['C']),
        (nodes['C'], nodes['E']),  # C + D -> E
        (nodes['D'], nodes['E']),
        (nodes['E'], nodes['G']),  # E + F -> G
        (nodes['F'], nodes['G']),
    ]

    # Converte para formato de tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Features iniciais para os nós (vetores one-hot para moléculas)
    node_features = torch.eye(len(nodes))  # Matriz identidade para features

    return Data(x=node_features, edge_index=edge_index)

# Modelo de GNN para otimização do grafo
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

# Função principal para treinamento
def train_model():
    # Cria o grafo de reações químicas
    data = create_reaction_graph()

    # Configurações do modelo
    input_dim = data.x.shape[1]
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
        loss = criterion(predictions, target_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Treinamento concluído!")
    return model

if __name__ == "__main__":
    trained_model = train_model()
