from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors

# Carregar modelo pré-treinado e tokenizer
def load_model():
    model_name = "gpt2"  # Substitua por um modelo ajustado, se disponível
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# Gerar reações químicas
def generate_reactions(model, tokenizer, input_reactions, max_length=100):
    inputs = tokenizer.encode(input_reactions, return_tensors="pt")
    outputs = model.generate(
        inputs, max_length=max_length, num_return_sequences=3, temperature=0.7
    )
    reactions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return reactions

# Validar reações geradas usando RDKit
def validate_reactions(reactions):
    valid_reactions = []
    for reaction in reactions:
        try:
            reactants, products = reaction.split("->")
            reactants_mols = [Chem.MolFromSmiles(smile.strip()) for smile in reactants.split(",")]
            products_mols = [Chem.MolFromSmiles(smile.strip()) for smile in products.split(",")]

            if all(reactants_mols) and all(products_mols):
                valid_reactions.append(reaction)
        except Exception as e:
            print(f"Invalid reaction format: {reaction}, Error: {e}")
    return valid_reactions

# Principal
if __name__ == "__main__":
    tokenizer, model = load_model()

    # Reações de exemplo (entrada)
    example_reactions = "C(C)O + O=O -> C(C)OO\n"  # SMILES de exemplo

    print("Gerando reações...")
    generated_reactions = generate_reactions(model, tokenizer, example_reactions)

    print("Reações geradas:")
    for reaction in generated_reactions:
        print(reaction)

    print("\nValidando reações...")
    valid_reactions = validate_reactions(generated_reactions)

    print("Reações válidas:")
    for reaction in valid_reactions:
        print(reaction)
