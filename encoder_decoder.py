import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem

# -------------------------------
# 1. Carregar Modelo Transformer
# -------------------------------
def load_molecular_model(model_name="t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# -------------------------------
# 2. Geração de Reações Químicas
# -------------------------------
def generate_simplifications(model, tokenizer, input_reactions, num_return_sequences=5):
    """
    Gera simplificações de reações químicas usando um modelo Transformer.
    """
    input_text = "Simplify: " + " | ".join(input_reactions)
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=100,
        num_beams=max(5, num_return_sequences),
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        do_sample=True
    )

    generated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    cleaned_generated = []
    for g in generated:
        g = g.replace("Simplify:", "").replace("Simplifier:", "").replace("Simplification:", "").replace("Simplifying:", "").strip()
        reactions = g.split("|")
        for reaction in reactions:
            reaction = reaction.strip()
            if "->" in reaction:
                cleaned_generated.append(reaction)

    return cleaned_generated

# -------------------------------
# 3. Validação das Reações Geradas
# -------------------------------
def validate_chemical_reactions(reactions):
    """
    Valida uma lista de reações químicas e remove aquelas que não seguem o formato correto.
    """
    valid_reactions = []

    for reaction in reactions:
        try:
            if "->" not in reaction:
                continue

            reactants, products = reaction.split("->")
            reactants = [r.strip() for r in reactants.split("+")]
            products = [p.strip() for p in products.split("+")]

            if not all(reactants) or not all(products):
                continue

            valid_reactions.append(reaction)

        except Exception as e:
            print(f"Erro ao validar reação '{reaction}': {e}")

    if not valid_reactions:
        print("Nenhuma reação válida foi gerada. Mantendo a reação original.")
        return reactions[:1]

    return valid_reactions

# -------------------------------
# 4. Função de Avaliação de Fitness
# -------------------------------
def fitness(simplified_reaction):
    """
    Calcula a aptidão da solução simplificada.
    Deve considerar o número de reações eliminadas e a correção química.
    """
    simplified_count = len(simplified_reaction.split("|")) 
    return -simplified_count

# -------------------------------
# 5. Operação de Cruzamento e Mutação
# -------------------------------
def crossover(parent1, parent2):
    """
    Realiza o cruzamento entre duas reações simplificadas.
    """
    split_point = random.randint(1, len(parent1) - 1)
    child = parent1[:split_point] + parent2[split_point:]
    return child

def mutate(reaction):
    """
    Aplica mutação na reação química.
    """
    if "->" in reaction:
        reactants, product = reaction.split("->")
        reactants = reactants.split("+")
        if len(reactants) > 1:
            random.shuffle(reactants) 
        reaction = f"{'+'.join(reactants)} -> {product}"
    return reaction

# -------------------------------
# 6. Algoritmo Evolutivo para Otimização
# -------------------------------
def genetic_algorithm(original_reactions, model, tokenizer, generations=50, population_size=10):
    """
    Algoritmo genético para encontrar a melhor simplificação de reações químicas.
    """
    population = generate_simplifications(model, tokenizer, original_reactions, num_return_sequences=population_size)
    
    population = validate_chemical_reactions(population)

    for generation in range(1, generations + 1):
        fitness_scores = [fitness(ind) for ind in population]

        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        parents = sorted_population[:2]

        if len(parents) < 2:
            parents = original_reactions[:2]

        new_population = parents[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = validate_chemical_reactions(new_population)

        best_fitness = max(fitness_scores)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return sorted_population[0]

# -------------------------------
# 7. Execução do Código
# -------------------------------
if __name__ == "__main__":
    tokenizer, model = load_molecular_model()

    reactions = [
        "A + B -> C",
        "C + D -> E",
        "E + F -> G",
    ]

    print("\nReações originais:")
    for r in reactions:
        print(r)

    best_solution = genetic_algorithm(reactions, model, tokenizer)

    print("\nMelhor solução simplificada:")
    print(best_solution)
