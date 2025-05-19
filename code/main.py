"""Main script."""
from plots import visualize_ingredients_choice, visualize_profits
from simulate import ChainSimulation

# Optimize recipes
torch_device = "cuda"
chain = ChainSimulation(torch_device)
# recipe = ['Horse S*men', 'Motor Oil', 'Paracetamol']
# recipe = [
#     'Cuke',
#     'Energy Drink',
#     'Horse S*men',
#     'Banana',
#     'Horse S*men',
# ]
# recipe = [
#     'Horse S*men',
#     'V*agra',
#     'Mega Bean',
#     'Donut',
#     'Iodine',
#     'Donut',
#     'Battery'
# ]
# recipe = ['Iodine', 'Flu Medicine', 'Horse S*men', 'Mouth Wash', 'Banana', 'Horse S*men', 'Iodine', 'Banana']
# result = chain.mix_recipe("OG Kush", recipe)
# print(
# f"""
# Receita: {recipe}
# Efeitos: {result['effects']}
# Custo: {result['cost']}
# Valor: {result['value']}
# """
# )
results_data, results_opt = chain.optimize_recipe(
    "OG Kush", num_simulations=50, batch_size=20_000, num_steps=10, T0=10.0)
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
