from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Descriptors

# Carregar modelo LLM adaptado para moléculas

def load_molecular_model(model_name="seyonec/ChemBERTa-zinc-base-v1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Geração de reações químicas

def generate_chemical_reactions(model, tokenizer, input_smiles, max_length=100, num_return_sequences=3):
    inputs = tokenizer.encode(input_smiles, return_tensors="pt")
    outputs = model.generate(
        inputs, max_length=max_length, num_return_sequences=num_return_sequences, temperature=0.7
    )
    reactions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return reactions

# Validação das reações geradas

def validate_chemical_reactions(reactions):
    valid_reactions = []
    for reaction in reactions:
        try:
            reactants, products = reaction.split("->")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(",")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(",")]

            if all(reactant_mols) and all(product_mols):
                valid_reactions.append(reaction)
        except Exception as e:
            print(f"Erro na reação {reaction}: {e}")
    return valid_reactions

# Otimização de reações químicas

def optimize_crns(reactions):
    reaction_map = {}
    for reaction in reactions:
        try:
            reactants, products = reaction.split("->")
            reactants = tuple(sorted(reactants.split(",")))
            products = products.split(",")

            # Mapear produto final para os reagentes iniciais
            for product in products:
                if product not in reaction_map:
                    reaction_map[product] = set(reactants)
                else:
                    reaction_map[product].update(reactants)
        except Exception as e:
            print(f"Erro ao processar reação {reaction}: {e}")

    # Simplificar as reações
    simplified_reactions = []
    for product, reactants in reaction_map.items():
        simplified_reactions.append(f"{', '.join(sorted(reactants))} -> {product}")

    return simplified_reactions

# Função principal
if __name__ == "__main__":
    # Carregar modelo
    tokenizer, model = load_molecular_model()

    # Entrada de exemplo
    input_reactions = [
        "a + b -> c",
        "c + d -> e",
        "e + f -> g",
    ]

    print("\nValidando reações químicas...")
    valid_reactions = validate_chemical_reactions(input_reactions)

    print("Reações válidas:")
    for reaction in valid_reactions:
        print(reaction)

    print("\nOtimização das CRNs...")
    optimized_reactions = optimize_crns(valid_reactions)

    print("Reações otimizadas:")
    for reaction in optimized_reactions:
        print(reaction)
