import random

# Nova Função de Fitness
def calculate_fitness(individual):
    unique_reactions = set(individual)
    intermediates = set()
    fitness = 0

    for reaction in unique_reactions:
        try:
            reactants, product = reaction.split("->")
            reactants = [r.strip() for r in reactants.split("+")]
            product = product.strip()

            intermediates.update(reactants)  # Adiciona intermediários
            intermediates.add(product)  # Adiciona produto final
        except ValueError:
            fitness -= 100  # Penaliza reações mal formatadas

    # Penaliza o número de intermediários e recompensa menos reações
    fitness -= len(unique_reactions) * 5
    fitness += len(intermediates) * 2
    return fitness

# Função de Simplificação Melhorada
def simplify_reaction(reaction1, reaction2):
    try:
        reactants1, product1 = reaction1.split("->")
        reactants2, product2 = reaction2.split("->")
        reactants1 = set(r.strip() for r in reactants1.split("+"))
        reactants2 = set(r.strip() for r in reactants2.split("+"))

        if product1.strip() in reactants2:
            new_reactants = reactants1 | (reactants2 - {product1.strip()})
            return f"{' + '.join(sorted(new_reactants))} -> {product2.strip()}"
        elif product2.strip() in reactants1:
            new_reactants = reactants2 | (reactants1 - {product2.strip()})
            return f"{' + '.join(sorted(new_reactants))} -> {product1.strip()}"
    except ValueError:
        pass
    return None

# Cruzamento Melhorado
# Cruzamento Melhorado com Verificação de Tamanho
def crossover(parent1, parent2):
    # Verifica se o tamanho permite divisão
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1 if random.random() < 0.5 else parent2  # Retorna uma cópia de um dos pais
    
    # Executa o crossover normalmente
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:split_point] + parent2[split_point:]
    return list(set(child))


# Mutação Melhorada
def mutate(individual, reactions):
    if random.random() < 0.5:
        # Simplificar duas reações aleatórias
        if len(individual) > 1:
            r1, r2 = random.sample(individual, 2)
            new_reaction = simplify_reaction(r1, r2)
            if new_reaction:
                individual.append(new_reaction)
    else:
        # Adicionar ou remover reações
        if random.random() < 0.5 and len(individual) > 1:
            individual.pop(random.randint(0, len(individual) - 1))
        else:
            individual.append(random.choice(reactions))
    return list(set(individual))

# Algoritmo Genético com Perturbação
def genetic_algorithm(reactions, generations=50, population_size=10):
    population = [random.sample(reactions, len(reactions)) for _ in range(population_size)]
    best_individual = None
    best_fitness = float("-inf")

    for generation in range(generations):
        population_fitness = [(ind, calculate_fitness(ind)) for ind in population]
        population_fitness.sort(key=lambda x: x[1], reverse=True)

        best_individual, best_fitness = population_fitness[0]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Se a população estagnar, introduz perturbação
        if generation > 10 and all(fitness == best_fitness for _, fitness in population_fitness):
            population = [random.sample(reactions, len(reactions)) for _ in range(population_size)]
            continue

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.choices([ind for ind, _ in population_fitness], k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, reactions)
            new_population.append(child)

        population = new_population + [ind for ind, _ in population_fitness[:population_size // 2]]

    return best_individual

# Reações de Entrada
reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G",
    "G + H -> I"
]




# Executar Algoritmo
best_solution = genetic_algorithm(reactions)
print("\nMelhor solução:")
if best_solution:
    for reaction in best_solution:
        print(reaction)
else:
    print("Nenhuma solução encontrada.")
