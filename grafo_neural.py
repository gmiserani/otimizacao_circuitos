import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Exemplo de um GNN simples para processar grafos de reações químicas
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(2, 16)  # 2 features por nó (exemplo)
        self.conv2 = GCNConv(16, 1)  # Saída única para simplificação

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# Exemplo de grafo de reações químicas
# Nós: [A, B, C, D, E, F, G]
# Arestas: [A->C, B->C, C->E, D->E, E->G, F->G]
x = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1]], dtype=torch.float)  # Features dos nós
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 2, 4, 4, 6, 6]], dtype=torch.long)  # Arestas

# Cria o grafo
data = Data(x=x, edge_index=edge_index)

# Instancia o modelo
model = SimpleGNN()

# Processa o grafo
output = model(data.x, data.edge_index)
print("Saída do GNN:", output)