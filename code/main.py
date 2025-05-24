"""Main script."""
import itertools
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

# Grid search over all parameter combinations
simulation_results = []
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

    # Store results
    simulation_results.append({
        'base_product': base_product,
        'recipe_size': recipe_size,
        'initial_temperature': initial_temperature,
        'results_data': to_cpu_recursive(results_data),
        'results_opt': to_cpu_recursive(results_opt)
    })
    print("Optimal Results:")
    print(f"# Recipe: {results_opt['recipe']}")
    print(f"# Effects: {results_opt['effects']}")
    print(f"# Cost: {results_opt['cost']}")
    print(f"# Value: {results_opt['value']}")
    print(f"# Profit: {results_opt['profit']}")
    print("-" * 40)

    # Clear memory
    del results_data, results_opt
    torch.cuda.empty_cache()


# Print each result in simulation_results
for result in simulation_results:
    # Plot results
    plot_final_step_ingredients_barplot(
        recipes=result['results_data']['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=result['base_product'],
        recipe_size=result['recipe_size'],
        initial_temperature=result['initial_temperature']
    )
    plot_ingredients_lineplot(
        recipes=result['results_data']['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=result['base_product'],
        recipe_size=result['recipe_size'],
        initial_temperature=result['initial_temperature']
    )
    plot_effects_lineplot(
        effects=result['results_data']['effects'],
        effects_name=chain.effects_df['effect_name'].tolist(),
        base_product=result['base_product'],
        recipe_size=result['recipe_size'],
        initial_temperature=result['initial_temperature']
    )
    plot_profit_lineplot(
        profits=result['results_data']['profits'],
        base_product=result['base_product'],
        recipe_size=result['recipe_size'],
        initial_temperature=result['initial_temperature']
    )
    plot_recipes_sankey(
        recipes=result['results_data']['recipes'],
        ingredients_name=chain.ingredients_df['ingredient_name'].tolist(),
        base_product=result['base_product'],
        recipe_size=result['recipe_size'],
        initial_temperature=result['initial_temperature']
    )
