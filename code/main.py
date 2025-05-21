"""Main script."""
from plots import visualize_ingredients_choice, visualize_profits
from simulate import ChainSimulation

# Optimize recipes
torch_device = "cuda"
chain = ChainSimulation(torch_device)
# recipe = [
#     'Banana',
#     'Gasoline',
#     'Paracetamol',
#     'Cuke',
#     'Mega Bean',
#     'Battery',
#     'Banana',
#     'Cuke',
# ]
# result = chain.mix_recipe("OG Kush", recipe)
# print(
# f"""
# Receita: {recipe}
# Efeitos: {result['effects']}
# Custo: {result['cost']}
# Valor: {result['value']}
# Lucro: {result['profit']}
# """
# )

results_data, results_opt = chain.optimize_recipe(
    base_product="OG Kush",
    objective_function='profit',
    num_simulations=1,
    batch_size=5,
    num_steps=10,
    T0=10.0
)
print(
f"""
OTIMIZADO:
Receita: {results_opt["recipe"]}
Efeitos: {results_opt["effects"]}
Custo: {results_opt["cost"]}
Valor: {results_opt["value"]}
Lucro: {results_opt["profit"]}
"""
)

# Plot simulation graphs
# visualize_ingredients_choice(
#     results_data['recipes'],
#     n_ingredients=chain.n_ingredients,
#     ingredients_df=chain.ingredients_df,
# )

# visualize_profits(profits=results_data['profits'])
