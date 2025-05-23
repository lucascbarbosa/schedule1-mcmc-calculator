"""Main script."""
from plots import (
    plot_effects_lineplot,
    plot_final_step_ingredients_barplot,
    plot_ingredients_lineplot,
    plot_profit_lineplot,
    plot_recipes_sankey
)
from simulate import ChainSimulation

# Optimize recipes
chain = ChainSimulation()
results_data, results_opt = chain.optimize_recipes(
    base_product="OG Kush",
    batch_size=10_000,
    n_steps=200,
    recipe_size=8,
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

# Plot results
# plot_final_step_ingredients_barplot(
#     recipes=results_data['recipes'],
#     ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
# )
# plot_ingredients_lineplot(
#     recipes=results_data['recipes'],
#     ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
# )

# plot_effects_lineplot(
#     effects=results_data['effects'],
#     effects_name=chain.effects_df['effect_name'].tolist(),
# )

# plot_profit_lineplot(
#     profits=results_data['profits'],
# )

# plot_recipes_sankey(
#     recipes=results_data['recipes'],
#     ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
# )