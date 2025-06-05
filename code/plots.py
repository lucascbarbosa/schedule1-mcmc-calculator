"""Script to plot optimization results."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from typing import List


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

    ingredients_frequency = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy().T  # shape: (n_ingredients, n_steps)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        ingredients_frequency,
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )
    ax.set_xlabel('Step', fontsize=16)
    ax.set_ylabel('Ingredient', fontsize=16)
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.set_yticks(np.arange(n_ingredients))
    ax.set_yticklabels(ingredients_name, fontsize=12)
    fig.colorbar(im, ax=ax, label='Relative frequency')
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_ingredients_heatmap.svg")
    fig.savefig(f"../plots/{base_product}_{recipe_size}_ingredients_heatmap.png")
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
    im = ax.imshow(
        effects_frequency,
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )
    ax.set_xlabel('Step', fontsize=16)
    ax.set_ylabel('Effect', fontsize=16)
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.set_yticks(np.arange(n_effects))
    ax.set_yticklabels(effects_name, fontsize=12)
    fig.colorbar(im, ax=ax, label='Relative frequency')
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_effects_heatmap.svg")
    fig.savefig(f"../plots/{base_product}_{recipe_size}_effects_heatmap.png")
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
    ax.set_xlabel('Step', fontsize=16)
    ax.set_ylabel('Profit', fontsize=16)
    ax.set_xlim(0.5, n_steps + 0.5)
    fig.tight_layout()
    fig.savefig(f"../plots/{base_product}_{recipe_size}_profit_lineplot.svg")
    fig.savefig(f"../plots/{base_product}_{recipe_size}_profit_lineplot.png")
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
    )
    fig.write_image(f"../plots/{base_product}_{recipe_size}_recipes_sankey.svg")
    fig.write_image(f"../plots/{base_product}_{recipe_size}_recipes_sankey.png")
    plt.close()

