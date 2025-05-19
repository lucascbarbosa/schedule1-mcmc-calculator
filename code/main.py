"""Main script."""
from plots import visualize_ingredients_choice, visualize_profits
from simulate import ChainSimulation

# Optimize recipes
chain = ChainSimulation(torch_device="cpu")
results_data, results_opt = chain.optimize_recipe(
    "OG Kush", num_simulations=1, batch_size=1_000, num_steps=10, T0=10.0)
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

# # Plot simulation graphs
# visualize_ingredients_choice(
#     results_data['recipes'],
#     n_ingredients=chain.n_ingredients
# )

# visualize_profits(profits=results_data['profits'])
