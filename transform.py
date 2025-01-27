from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def simplify_reactions_with_bart(reactions):
    """
    Simplifica reações químicas usando o BART.
    
    :param reactions: Lista de reações químicas no formato "A + B -> C".
    :return: Lista de reações simplificadas.
    """
    # Carregue o modelo e o tokenizador
    model_name = "facebook/bart-large"  # Modelo genérico
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize as reações químicas
    inputs = tokenizer(reactions, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Gere previsões
    outputs = model.generate(**inputs)
    simplified_reactions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return simplified_reactions

# Exemplo de uso
reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]

simplified = simplify_reactions_with_bart(reactions)
print("Reações simplificadas (usando BART):")
for reaction in simplified:
    print(reaction)