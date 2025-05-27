"""Main script."""
import itertools
import pandas as pd
import torch
from plots import (
    plot_effects_lineplot,
    plot_final_step_ingredients_barplot,
    plot_ingredients_lineplot,
    plot_profit_lineplot,
    plot_recipes_sankey,
    plot_profit_barplot,
)
from simulate import ChainSimulation


def to_cpu_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu_recursive(x) for x in obj]
    else:
        return obj


# Optimize recipes
chain = ChainSimulation()

simulation_data = {
    # 'base_product': chain.products_df['product_name'].to_numpy(),
    # 'recipe_size': [7, 8, 9, 10],
    'base_product': ['Cocaine'],
    'recipe_size': [15],
    'batch_size': [10_000],
    'n_steps': [1000],
    'initial_temperature': [100.0],
    'alpha': [0.99],
}

results_df = []
for (
    base_product,
    recipe_size,
    batch_size,
    n_steps,
    initial_temperature,
    alpha
) in itertools.product(
    simulation_data['base_product'],
    simulation_data['recipe_size'],
    simulation_data['batch_size'],
    simulation_data['n_steps'],
    simulation_data['initial_temperature'],
    simulation_data['alpha'],
):
    print("Running:")
    print(f"# Base Product: {base_product}")
    print(f"# Recipe Size: {recipe_size}")
    print(f"# Steps: {n_steps}")
    print(f"# T0: {initial_temperature}")
    print(f"# Alpha: {alpha}")
    results_data, results_opt = chain.optimize_recipes(
        base_product=base_product,
        batch_size=batch_size,
        n_steps=n_steps,
        recipe_size=recipe_size,
        alpha=alpha,
        initial_temperature=initial_temperature,
    )
    print("Optimal Results:")
    print(f"# Recipe: {results_opt['recipe']}")
    print(f"# Effects: {results_opt['effects']}")
    print(f"# Cost: {results_opt['cost']}")
    print(f"# Value: {results_opt['value']}")
    print(f"# Profit: {results_opt['profit']}")
    print("-" * 40)

    results_df.append(
        {
            'Base Product': base_product,
            'Recipe Size': recipe_size,
            'Steps': n_steps,
            'T0': initial_temperature,
            'Alpha': alpha,
            'Recipe': results_opt['recipe'],
            'Effects': results_opt['effects'],
            'Cost': results_opt['cost'],
            'Value': results_opt['value'],
            'Profit': results_opt['profit'],
        }
    )

    # Move results to CPU
    results_data = to_cpu_recursive(results_data)
    results_opt = to_cpu_recursive(results_opt)

    # Plot results immediately
    plot_final_step_ingredients_barplot(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_ingredients_lineplot(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_effects_lineplot(
        effects=results_data['effects'],
        effects_name=chain.effects_df['effect_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_profit_lineplot(
        profits=results_data['profits'],
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_recipes_sankey(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )

    # Clear memory
    del results_data, results_opt
    torch.cuda.empty_cache()

results_df = pd.DataFrame(results_df)
results_df.to_excel('../results/optimization_results.xlsx', index=False)

# Plot results heatmap
plot_profit_barplot(results_df)
