from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

# Configuração do modelo molecular pré-treinado
def load_molecular_model(model_name="seyonec/ChemBERTa-zinc-base-v1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Gerar simplificações iniciais com o modelo pré-treinado
def generate_simplifications(model, tokenizer, reactions, max_length=100, num_return_sequences=10):
    """
    Gera simplificações iniciais para as reações fornecidas.

    Args:
        model: Modelo de geração (Transformer).
        tokenizer: Tokenizer associado ao modelo.
        reactions: Lista de reações químicas como entrada.
        max_length: Comprimento máximo das sequências geradas.
        num_return_sequences: Número de sequências geradas para cada entrada.

    Returns:
        Lista de simplificações geradas.
    """
    inputs = tokenizer(reactions, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        do_sample=True,  # Habilita amostragem para gerar múltiplas sequências
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Avaliar a validade química (simples validação de formato)
def validate_reaction(reaction):
    try:
        reactants, products = reaction.split("->")
        reactants = [r.strip() for r in reactants.split("+")]
        products = [p.strip() for p in products.split("+")]
        return all(reactants) and all(products)
    except Exception:
        return False

# Funções do algoritmo evolutivo
def crossover(parent1, parent2):
    split_point = random.randint(1, len(parent1.split("->")[0]))
    child1 = parent1[:split_point] + parent2[split_point:]
    child2 = parent2[:split_point] + parent1[split_point:]
    return child1, child2

def mutate(reaction):
    if "->" in reaction:
        reactants, products = reaction.split("->")
        reactants = reactants.split("+")
        random.shuffle(reactants)
        return f"{'+'.join(reactants)} -> {products.strip()}"
    return reaction

# Função de avaliação (fitness)
def fitness(reaction):
    return -len(reaction) if validate_reaction(reaction) else float('inf')

# Algoritmo Evolutivo
def genetic_algorithm(reactions, model, tokenizer, generations=50, population_size=10):
    # Geração inicial com simplificações do modelo
    population = generate_simplifications(model, tokenizer, reactions, num_return_sequences=population_size)

    for generation in range(generations):
        population = [r for r in population if validate_reaction(r)]  # Filtrar reações inválidas
        population = sorted(population, key=fitness)  # Ordenar pela melhor fitness

        next_generation = population[:2]  # Elitismo: manter os 2 melhores

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population[:5], 2)  # Seleção dos melhores
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])

        population = next_generation

        # Melhor fitness da geração atual
        best_fitness = fitness(population[0])
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        if best_fitness == -1:  # Parada antecipada se encontrar a melhor solução
            break

    return population[0]  # Melhor solução encontrada

# Função principal
if __name__ == "__main__":
    # Carregar modelo molecular
    tokenizer, model = load_molecular_model()

    # Exemplo de rede de reações
    reactions = [
        "A + B -> C",
        "C + D -> E",
        "E + F -> G"
    ]

    print("Reações originais:")
    for r in reactions:
        print(r)

    # Aplicar algoritmo evolutivo para simplificação
    best_solution = genetic_algorithm(reactions, model, tokenizer)

    print("\nMelhor solução simplificada:")
    print(best_solution)
