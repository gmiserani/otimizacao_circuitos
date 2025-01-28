import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(2, 16) 
        self.conv2 = GCNConv(16, 1) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        edge_scores = torch.sigmoid(x)
        return edge_scores[edge_index[0]]

def simplify_reactions(x, edge_index, edge_scores, threshold=0.5):
    """
    Consolida a rede química com base nos scores das arestas.
    """
    mask = edge_scores.squeeze() > threshold
    relevant_edges = edge_index[:, mask]
    return relevant_edges.t().tolist()

x = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [1, 1]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 2, 4, 4, 6, 6]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

model = SimpleGNN()

edge_scores = model(data.x, data.edge_index)

print("Scores das arestas:", edge_scores.squeeze().tolist())

simplified_edges = simplify_reactions(data.x, data.edge_index, edge_scores, threshold=0.1)
print("Arestas simplificadas:", simplified_edges)
