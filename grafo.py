import networkx as nx

def simplify_crn(reactions):
    """
    Simplifica uma rede de reações químicas (CRN) eliminando intermediários desnecessários.
    
    :param reactions: Lista de reações químicas no formato "A + B -> C".
    :return: Lista de reações simplificadas.
    """
    G = nx.DiGraph()
    
    for reaction in reactions:
        reactants, products = reaction.split('->')
        reactants = reactants.split('+')
        products = products.split('+')
        
        for r in reactants:
            for p in products:
                G.add_edge(r.strip(), p.strip())
    
    independent_paths = []
    for node in G.nodes:
        if G.in_degree(node) == 0:
            path = list(nx.dfs_preorder_nodes(G, source=node))
            independent_paths.append(path)
    
    simplified_reactions = []
    for path in independent_paths:
        if len(path) > 1:
            simplified_reactions.append(f"{path[0]} + {path[1]} -> {path[-1]}")
        else:
            simplified_reactions.append(f"{path[0]} -> {path[0]}")
    
    return simplified_reactions

reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]

simplified = simplify_crn(reactions)
print("Reações simplificadas (abordagem em grafos):")
for reaction in simplified:
    print(reaction)