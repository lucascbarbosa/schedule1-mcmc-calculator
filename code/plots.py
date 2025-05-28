"""Script to plot optimization results."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from typing import List


def plot_final_step_ingredients_barplot(
    recipes: torch.Tensor,
    ingredients_name: List[str],
    base_product: str,
    recipe_size: int,
):
    """Plot ingredients relative frequency per recipe position in final step."""
    # Get recipe at last step
    recipe = recipes[-1].int()
    recipe_size = recipe.shape[0]
    n_ingredients = len(ingredients_name)

    # Count ingredient occurrences for last step per recipe position
    ingredients_count = torch.stack([
        torch.bincount(
            recipe[i, :].reshape(-1), minlength=n_ingredients + 1
        )
        for i in range(recipe_size)
    ])[:, :-1]

    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum()
    ).cpu().numpy()  # shape: (recipe_size, n_ingredients)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    recipe_positions = np.arange(1, recipe_size + 1)
    bottom = np.zeros(recipe_size)
    for i, name in enumerate(ingredients_name):
        ax.bar(
            recipe_positions,
            ingredients_proportion[:, i],
            bottom=bottom,
            label=name
        )
        bottom += ingredients_proportion[:, i]

    ax.set_xlabel("Recipe Position")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(
        "Ingredient relative frequency per recipe position (at last step)\n"
        f"Base Product: {base_product}, Recipe size: {recipe_size}",
        fontsize=14,
        loc='center'
    )
    ax.set_xticks(recipe_positions)
    ax.set_xticklabels([str(i) for i in recipe_positions])
    ax.legend(title="Ingredient", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_ingredients_barplot.svg")
    plt.close()


def plot_ingredients_heatmap(
    recipes: torch.Tensor,
    ingredients_name: List[str],
    base_product: str,
    recipe_size: int,
):
    """Plot ingredients relative frequency per step as a heatmap."""
    recipes = recipes.int()
    n_steps = recipes.shape[0]
    n_ingredients = len(ingredients_name)

    # Flatten all batches for each step and count ingredient occurrences
    ingredients_count = torch.stack([
        torch.bincount(
            recipes[step].reshape(-1), minlength=n_ingredients + 1
        )
        for step in range(n_steps)
    ])[:, :-1]

    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy().T  # shape: (n_ingredients, n_steps)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(ingredients_proportion, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ingredient')
    ax.set_title(
        "Relative frequency of ingredients per simulation step (heatmap)\n"
        f"Base Product: {base_product}, Recipe size: {recipe_size}",
        fontsize=14,
        loc='center'
    )
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.set_yticks(np.arange(n_ingredients))
    ax.set_yticklabels(ingredients_name)
    fig.colorbar(im, ax=ax, label='Relative frequency')
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_ingredients_heatmap.svg")
    plt.close()


def plot_effects_heatmap(
    effects: torch.Tensor,
    effects_name: List[str],
    base_product: str,
    recipe_size: int,
):
    """Plot the relative frequency of each effect as a heatmap."""
    n_steps, n_effects, batch_size = effects.shape

    # Calculate mean presence of each effect per step
    effects_frequency = torch.zeros(
        (n_steps, n_effects),
        dtype=torch.float32
    )
    for step in range(n_steps):
        effects_frequency[step] = (
            effects[step].sum(dim=1).float() /
            batch_size
        )

    # Transpose to shape (n_effects, n_steps) for plotting
    effects_frequency = effects_frequency.cpu().numpy().T

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(effects_frequency, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effect')
    ax.set_title(
        "Relative frequency of effects per simulation step (heatmap)\n"
        f"Base Product: {base_product}, Recipe size: {recipe_size}",
        fontsize=14,
        loc='center'
    )
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.set_yticks(np.arange(n_effects))
    ax.set_yticklabels(effects_name)
    fig.colorbar(im, ax=ax, label='Relative frequency')
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_effects_heatmap.svg")
    plt.close()


def plot_profit_lineplot(
    profits: torch.Tensor,
    base_product: str,
    recipe_size: int,
):
    """Plot mean profit per step with 95% confidence interval as a line plot."""
    n_steps = profits.shape[0]
    profits_np = profits.cpu().numpy()  # shape: (n_steps, n_samples)

    steps = np.arange(1, n_steps + 1)
    means = profits_np.mean(axis=1)
    stds = profits_np.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, means, label='Mean Profit', color='blue')
    ax.fill_between(
        steps, means - 3 * stds, means + 3 * stds, color='blue', alpha=0.2
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Profit')
    ax.set_title(
        "Mean profit per step with confidence interval per simulation step\n"
        f"Base Product: {base_product}, Recipe size: {recipe_size}",
        fontsize=14,
        loc='center'
    )
    ax.set_xlim(0.5, n_steps + 0.5)
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_profit_lineplot.svg")
    plt.close()


def plot_recipes_sankey(
    recipes: torch.Tensor,
    ingredients_name: list,
    base_product: str,
    recipe_size: int,
):
    """Plot a Sankey diagram for the last step of recipes.

    Each node is an ingredient at a recipe position.
    Each link connects ingredient at position i to ingredient at position i+1,
    and its value is the number of recipes that contain both ingredients at
    those positions.

    """
    # Use only the last step
    last_step = recipes[-1]  # shape: (recipe_size, batch_size)
    recipe_size, n_recipes = last_step.shape
    n_ingredients = len(ingredients_name)

    # Build node labels: "Ingredient (pos i)"
    node_labels = []
    node_map = {}
    for pos in range(recipe_size):
        for ing_id in range(n_ingredients):
            label = ingredients_name[ing_id]
            node_map[(pos, ing_id)] = len(node_labels)
            node_labels.append(label)

    # Count links between positions i and i+1
    link_values = {}
    for sim in range(n_recipes):
        for pos in range(recipe_size - 1):
            ing_i = last_step[pos, sim].item()
            ing_j = last_step[pos + 1, sim].item()
            if ing_i < n_ingredients and ing_j < n_ingredients:
                source = node_map[(pos, ing_i)]
                target = node_map[(pos + 1, ing_j)]
                link_values[
                    (source, target)
                ] = link_values.get((source, target), 0) + 1

    if not link_values:
        print("No links to plot.")
        return

    sources, targets, values = zip(
        *[(s, t, v) for (s, t), v in link_values.items()]
    )
    values = np.array(values)
    values_scaled = values / values.max()
    colors = [
        f'rgba({int(255 * (1 - v))},{int(255 * (1 - v))},{int(255 * (1 - v))}, 0.5)'
        for v in values_scaled
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=node_labels,
                pad=15,
                thickness=15,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors,
            )
        ),
    )
    fig.update_layout(
        width=1200,
        height=600,
        title_text=(
            "Sankey Diagram of Ingredients per recipe position (at last step)\n"
            f"Base Product: {base_product}, Recipe size: {recipe_size}"
        ),
        font_size=12
    )
    fig.write_image(f"../plots/{base_product}_{recipe_size}_recipes_sankey.svg")
    plt.close()


def plot_profit_barplot(results_df: pd.DataFrame):
    """Plot profit barplot for each base product and recipe size."""
    plt.figure(figsize=(10, 6))
    recipe_sizes = sorted(results_df['Recipe Size'].unique())
    base_products = results_df['Base Product'].unique()
    bar_width = 0.8 / len(recipe_sizes)
    x = np.arange(len(base_products))

    for idx, recipe_size in enumerate(recipe_sizes):
        profits = []
        for bp in base_products:
            row = results_df[
                (results_df['Base Product'] == bp) &
                (results_df['Recipe Size'] == recipe_size)
            ]
            if not row.empty:
                profits.append(row['Profit'].to_numpy()[0])
            else:
                profits.append(0)
        plt.bar(
            x + idx * bar_width,
            profits,
            width=bar_width,
            label=f"Recipe Size {recipe_size}"
        )

    plt.xlabel('Base Product')
    plt.ylabel('Profit')
    plt.title('Profit by Base Product and Recipe Size')
    plt.xticks(x + bar_width * (len(recipe_sizes) - 1) / 2, base_products)
    plt.legend(title='Recipe Size')
    plt.tight_layout()
    plt.savefig('../plots/profit_barplot.svg')
    plt.close()
