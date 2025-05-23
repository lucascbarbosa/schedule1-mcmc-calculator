"""Main script."""
from plots import (
    plot_ingredients_heatmap,
    plot_effects_heatmap,
    plot_profits_boxplot,
)
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
    batch_size=15_000,
    num_steps=7,
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

# Plot results
plot_ingredients_heatmap(
    recipes=results_data['recipes'],
    ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
)

plot_effects_heatmap(
    effects=results_data['effects'],
    effects_name=chain.effects_df['effect_name'].tolist(),
)

plot_profits_boxplot(
    profits=results_data['profits'],
)
