"""Main script."""
from plots import visualize_ingredients_choice, visualize_profits
from simulate import ChainSimulation


# Optimize recipes
chain = ChainSimulation()
results_data, results_opt = chain.optimize_recipe(
    "OG Kush", batch_size=100, num_steps=8, T0=1.0)
print(
f"""
OTIMIZADO:
Receita: {results_opt["recipe"]}
Efeitos: {results_opt["effects"]}
Custo: {results_opt["cost"]}
Valor: {results_opt["value"]}
Profit: {results_opt["profit"]}
"""
)

# Plot simulation graphs
visualize_ingredients_choice(
    results_data['recipes'],
    n_ingredients=chain.n_ingredients
)

visualize_profits(profits=results_data['profits'])
