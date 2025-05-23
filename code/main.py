"""Main script."""
from plots import (
    plot_ingredients_heatmap,
    plot_effects_heatmap,
    plot_profits_boxplot,
    plot_recipes_sankey
)
from simulate import ChainSimulation

# Optimize recipes
chain = ChainSimulation()
results_data, results_opt = chain.optimize_recipes(
    base_product="OG Kush",
    batch_size=200_000,
    n_steps=100,
    initial_temperature=1.0,
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

# # Plot results
# plot_ingredients_heatmap(
#     recipes=results_data['recipes'],
#     ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
# )

# plot_effects_heatmap(
#     effects=results_data['effects'],
#     effects_name=chain.effects_df['effect_name'].tolist(),
# )

# plot_profits_boxplot(
#     profits=results_data['profits'],
# )

# plot_recipes_sankey(
#     profits=results_data['profits'],
#     recipes=results_data['recipes'],
#     ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
# )