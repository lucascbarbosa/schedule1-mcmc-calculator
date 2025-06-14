"""Script to optimize recipe."""

import itertools
import pandas as pd
import torch
from plots import (
    plot_effects_heatmap,
    plot_ingredients_heatmap,
    plot_profit_lineplot,
    plot_recipes_sankey,
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

simulation_parameters = {
    'base_product': chain.products_df['product_name'].to_numpy(),
    'recipe_size': [7, 8, 9, 10],
    'n_simulations': [10_000],
    'n_steps': [600],
    'initial_temperature': [100.0],
    'alpha': [0.99],
}

results_df = []
for (
    base_product,
    recipe_size,
    n_simulations,
    n_steps,
    initial_temperature,
    alpha,
) in itertools.product(
    simulation_parameters["base_product"],
    simulation_parameters["recipe_size"],
    simulation_parameters["n_simulations"],
    simulation_parameters["n_steps"],
    simulation_parameters["initial_temperature"],
    simulation_parameters["alpha"],
):
    print("Running:")
    print(f"# Base Product: {base_product}")
    print(f"# Recipe Size: {recipe_size}")
    print(f"# Steps: {n_steps}")
    print(f"# T0: {initial_temperature}")
    print(f"# Alpha: {alpha}")
    results_data, results_opt = chain.optimize_recipes(
        base_product=base_product,
        n_simulations=n_simulations,
        n_steps=n_steps,
        recipe_size=recipe_size,
        alpha=alpha,
        initial_temperature=initial_temperature,
    )
    print("\nOptimal Results:")
    print(f"# Recipe: {results_opt['recipe']}")
    print(f"# Effects: {results_opt['effects']}")
    print(f"# Cost: {results_opt['cost']}")
    print(f"# Value: {results_opt['value']}")
    print(f"# Profit: {results_opt['profit']}")
    print("-" * 40)

    results_df.append(
        {
            "Base Product": base_product,
            "Initial Recipe Size": recipe_size,
            "Steps": n_steps,
            "T0": initial_temperature,
            "Alpha": alpha,
            "Recipe": results_opt["recipe"],
            "Effects": results_opt["effects"],
            "Cost": results_opt["cost"],
            "Value": results_opt["value"],
            "Profit": results_opt["profit"],
        }
    )

    # Move results to CPU
    results_data = to_cpu_recursive(results_data)
    results_opt = to_cpu_recursive(results_opt)

    # Plot results immediately
    plot_ingredients_heatmap(
        recipes=results_data["recipes"],
        ingredients_name=chain.ingredients_df["ingredient_name"].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_effects_heatmap(
        effects=results_data["effects"],
        effects_name=chain.effects_df["effect_name"].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_profit_lineplot(
        profits=results_data["profits"],
        base_product=base_product,
        recipe_size=recipe_size,
    )
    plot_recipes_sankey(
        recipes=results_data["recipes"],
        ingredients_name=chain.ingredients_df["ingredient_name"].tolist(),
        base_product=base_product,
        recipe_size=recipe_size,
    )

    # Clear memory
    del results_data, results_opt
    torch.cuda.empty_cache()

results_df = pd.DataFrame(results_df)
results_df.to_excel("../results/optimization_results.xlsx", index=False)
