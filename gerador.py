import random

def generate_synthetic_crn(num_molecules=10, num_reactions=5):
    """
    Gera uma rede de reações químicas sintética (CRN).

    Args:
        num_molecules (int): Número de moléculas únicas na CRN.
        num_reactions (int): Número de reações na CRN.

    Returns:
        dict: CRN com 'nodes' e 'edges'.
    """
    molecules = [chr(65 + i) for i in range(num_molecules)]  # ['A', 'B', ..., 'J']

    edges = []
    for _ in range(num_reactions):
        num_reactants = random.randint(2, 3)
        reactants = random.sample(molecules, num_reactants)
        product = random.choice([m for m in molecules if m not in reactants])
        edges.append((reactants, product))

    return {
        'nodes': molecules,
        'edges': edges
    }

def generate_dataset(num_samples=100, max_molecules=15, max_reactions=10):
    """
    Gera um conjunto de dados sintéticos de CRNs.

    Args:
        num_samples (int): Número de CRNs a serem geradas.
        max_molecules (int): Número máximo de moléculas por CRN.
        max_reactions (int): Número máximo de reações por CRN.

    Returns:
        list: Lista de CRNs sintéticas.
    """
    dataset = []
    for _ in range(num_samples):
        num_molecules = random.randint(5, max_molecules)
        num_reactions = random.randint(3, max_reactions)
        crn = generate_synthetic_crn(num_molecules, num_reactions)
        dataset.append(crn)
    return dataset
