import networkx as nx

def simplify_crn_with_astar(reactions):
    """
    Simplifica uma rede de reações químicas usando o algoritmo A*.
    
    :param reactions: Lista de reações químicas no formato "A + B -> C".
    :return: Lista de reações simplificadas.
    """
    # Cria um grafo direcionado
    G = nx.DiGraph()
    
    # Adiciona as reações ao grafo
    for reaction in reactions:
        reactants, products = reaction.split('->')
        reactants = reactants.split('+')
        products = products.split('+')
        
        for r in reactants:
            for p in products:
                G.add_edge(r.strip(), p.strip(), weight=1)  # Peso 1 para todas as arestas
    
    # Encontra o caminho mais curto entre os reagentes iniciais e os produtos finais
    simplified_reactions = []
    initial_reactants = [node for node in G.nodes if G.in_degree(node) == 0]
    final_products = [node for node in G.nodes if G.out_degree(node) == 0]
    
    for r in initial_reactants:
        for p in final_products:
            try:
                path = nx.astar_path(G, r, p)
                if len(path) > 1:
                    simplified_reactions.append(f"{path[0]} + {path[1]} -> {path[-1]}")
            except nx.NetworkXNoPath:
                continue
    
    return simplified_reactions

# Exemplo de uso
reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]
# Saída esperada: A + B -> G
reactions = [
    "A + B -> C",
    "D + E -> F",
    "F + G -> H"
]
# Saída esperada: A + B -> C e D + E -> H

reactions = [
    "A + B -> C",
    "C + D -> E",
    "A + F -> G",
    "G + D -> E"
]
# Saída esperada: A + B -> E e A + F -> E

reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> A"  # Ciclo: A -> C -> E -> A
]
# Saída esperada: A + B -> E (ciclo detectado)

reactions = [
    "A + B -> C",
    "C + D -> E",
    "C + F -> G"
]
# Saída esperada: A + B -> E e A + B -> G

simplified = simplify_crn_with_astar(reactions)
print("Reações simplificadas (usando A*):")
for reaction in simplified:
    print(reaction)