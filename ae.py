import random

# Definindo uma reação química como uma classe
class Reacao:
    def __init__(self, reagentes, produtos):
        self.reagentes = reagentes
        self.produtos = produtos

    def __repr__(self):
        return f"{' + '.join(self.reagentes)} -> {' + '.join(self.produtos)}"

# Função para simplificar uma rede de reações
def simplificar_rede(rede):
    simplificada = []
    intermediarios = set()

    # Identifica todos os produtos intermediários
    for reacao in rede:
        intermediarios.update(reacao.produtos)

    # Filtra as reações que não produzem intermediários
    for reacao in rede:
        if not any(produto in intermediarios for produto in reacao.reagentes):
            simplificada.append(reacao)
        else:
            # Se a reação usa intermediários, tenta encontrar uma reação simplificada
            novos_reagentes = []
            for reagente in reacao.reagentes:
                if reagente in intermediarios:
                    # Encontra a reação que produz esse intermediário
                    for r in rede:
                        if reagente in r.produtos:
                            novos_reagentes.extend(r.reagentes)
                else:
                    novos_reagentes.append(reagente)
            # Cria uma nova reação simplificada
            nova_reacao = Reacao(novos_reagentes, reacao.produtos)
            simplificada.append(nova_reacao)

    return simplificada

# Função para gerar uma população inicial de redes de reações
def gerar_populacao_inicial(reacoes, tamanho_populacao):
    populacao = []
    for _ in range(tamanho_populacao):
        # Seleciona aleatoriamente algumas reações para formar uma rede
        rede = random.sample(reacoes, random.randint(1, len(reacoes)))
        populacao.append(rede)
    return populacao

# Função de fitness para avaliar quão boa é uma rede de reações
def fitness(rede):
    rede_simplificada = simplificar_rede(rede)
    # Quanto menos reações, melhor
    return -len(rede_simplificada)

# Função para selecionar os melhores indivíduos da população
def selecionar(populacao, fitness, num_selecionados):
    # Ordena a população com base no fitness
    populacao_ordenada = sorted(populacao, key=fitness, reverse=True)
    return populacao_ordenada[:num_selecionados]

# Função para cruzar duas redes de reações
def cruzar(rede1, rede2):
    # Combina as reações das duas redes
    nova_rede = list(set(rede1 + rede2))
    return nova_rede

# Função para mutar uma rede de reações
def mutar(rede, reacoes):
    # Adiciona ou remove uma reação aleatoriamente
    if random.random() < 0.5:
        if rede:
            rede.remove(random.choice(rede))
    else:
        rede.append(random.choice(reacoes))
    return rede

# Algoritmo genético principal
def algoritmo_genetico(reacoes, tamanho_populacao, num_geracoes, taxa_mutacao):
    populacao = gerar_populacao_inicial(reacoes, tamanho_populacao)
    for geracao in range(num_geracoes):
        print(f"Geração {geracao + 1}")
        populacao = selecionar(populacao, fitness, tamanho_populacao // 2)
        nova_populacao = []
        while len(nova_populacao) < tamanho_populacao:
            pai1 = random.choice(populacao)
            pai2 = random.choice(populacao)
            filho = cruzar(pai1, pai2)
            if random.random() < taxa_mutacao:
                filho = mutar(filho, reacoes)
            nova_populacao.append(filho)
        populacao = nova_populacao
    melhor_rede = selecionar(populacao, fitness, 1)[0]
    return simplificar_rede(melhor_rede)

# Exemplo de uso
reacoes = [
    Reacao(["A", "B"], ["C"]),
    Reacao(["C", "D"], ["E"]),
    Reacao(["E", "F"], ["G"]),
    Reacao(["D", "E"], ["F"]),
    Reacao(["F", "G"], ["H"])
]

melhor_rede = algoritmo_genetico(reacoes, tamanho_populacao=20, num_geracoes=50, taxa_mutacao=0.1)
print("Melhor rede simplificada:")
for reacao in melhor_rede:
    print(reacao)