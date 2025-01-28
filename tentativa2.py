from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem

# Carregar modelo LLM adaptado para moléculas
def load_molecular_model(model_name="seyonec/ChemBERTa-zinc-base-v1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Validação das reações químicas
def validate_chemical_reactions(reactions):
    valid_reactions = []
    for reaction in reactions:
        try:
            reactants, products = reaction.split("->")
            reactants = reactants.split("+")
            products = products.split("+")

            # Verificar se são SMILES ou placeholders genéricos
            def is_valid_smiles_or_placeholder(compounds):
                return all(
                    Chem.MolFromSmiles(comp.strip()) or comp.isalnum()
                    for comp in compounds
                )

            if is_valid_smiles_or_placeholder(reactants) and is_valid_smiles_or_placeholder(products):
                valid_reactions.append(reaction)
            else:
                print(f"Reação inválida: {reaction}")
        except Exception as e:
            print(f"Erro na reação {reaction}: {e}")
    return valid_reactions

# Otimização de reações químicas
def optimize_crns(reactions):
    reaction_map = {}
    for reaction in reactions:
        try:
            reactants, products = reaction.split("->")
            reactants = tuple(sorted(reactants.split("+")))
            products = products.split("+")

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
        simplified_reactions.append(f"{' + '.join(sorted(reactants))} -> {product}")

    return simplified_reactions

# Função principal
if __name__ == "__main__":
    # Carregar modelo
    tokenizer, model = load_molecular_model()

    # Entrada de exemplo (genérica e SMILES)
    input_reactions = [
        "A + B -> C",
        "C + D -> E",
        "E + F -> G",
    ]

    print("\nValidando reações químicas...")
    valid_reactions = validate_chemical_reactions(input_reactions)

    print("\nReações válidas:")
    for reaction in valid_reactions:
        print(reaction)

    print("\nOtimização das CRNs...")
    optimized_reactions = optimize_crns(valid_reactions)

    print("\nReações otimizadas:")
    for reaction in optimized_reactions:
        print(reaction)
