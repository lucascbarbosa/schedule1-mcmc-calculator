"""Main script."""
import itertools
import pandas as pd
import torch
from plots import (
    plot_effects_lineplot,
    plot_final_step_ingredients_barplot,
    plot_ingredients_lineplot,
    plot_profit_lineplot,
    plot_recipes_sankey
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
    'base_product': chain.products_df['product_name'].to_numpy(),
    'recipe_size': [7, 8, 9, 10],
    'initial_temperature': [5.0, 10.0, 20.0, 50.0],
}

results_df = []
for base_product, recipe_size, initial_temperature in itertools.product(
    simulation_data['base_product'],
    simulation_data['recipe_size'],
    simulation_data['initial_temperature'],
):
    print("Running:")
    print(f"# Base Product: {base_product}")
    print(f"# Recipe Size: {recipe_size}")
    print(f"# Initial Temperature: {initial_temperature}")
    results_data, results_opt = chain.optimize_recipes(
        base_product=base_product,
        batch_size=10_000,
        n_steps=300,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature,
    )

    results_df.append(
        {
            'Base Product': base_product,
            'Recipe Size': recipe_size,
            'Initial Temperature': initial_temperature,
            'Recipe': results_opt['recipe'],
            'Effects': results_opt['effects'],
            'Cost': results_opt['cost'],
            'Value': results_opt['value'],
            'Profit': results_opt['profit'],
        }
    )
    print("Optimal Results:")
    print(f"# Recipe: {results_opt['recipe']}")
    print(f"# Effects: {results_opt['effects']}")
    print(f"# Cost: {results_opt['cost']}")
    print(f"# Value: {results_opt['value']}")
    print(f"# Profit: {results_opt['profit']}")
    print("-" * 40)

    # Move results to CPU
    results_data = to_cpu_recursive(results_data)
    results_opt = to_cpu_recursive(results_opt)

    # Plot results immediately
    plot_final_step_ingredients_barplot(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature
    )
    plot_ingredients_lineplot(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature
    )
    plot_effects_lineplot(
        effects=results_data['effects'],
        effects_name=chain.effects_df['effect_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature
    )
    plot_profit_lineplot(
        profits=results_data['profits'],
        base_product=base_product,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature
    )
    plot_recipes_sankey(
        recipes=results_data['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
        initial_temperature=initial_temperature
    )

    # Clear memory
    del results_data, results_opt
    torch.cuda.empty_cache()

results_df = pd.DataFrame(results_df)
results_df.to_excel('../results/optimization_results.xlsx')