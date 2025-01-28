from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def simplify_reactions_with_bart(reactions):
    """
    Simplifica reações químicas usando o BART.
    
    :param reactions: Lista de reações químicas no formato "A + B -> C".
    :return: Lista de reações simplificadas.
    """
    model_name = "facebook/bart-large" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    inputs = tokenizer(reactions, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    outputs = model.generate(**inputs)
    simplified_reactions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return simplified_reactions

reactions = [
    "A + B -> C",
    "C + D -> E",
    "E + F -> G"
]

simplified = simplify_reactions_with_bart(reactions)
print("Reações simplificadas (usando BART):")
for reaction in simplified:
    print(reaction)