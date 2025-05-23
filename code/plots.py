"""Script to visualize plots of optimization."""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from typing import List


def plot_ingredients_heatmap(
    recipes: torch.Tensor,
    ingredients_name: List[str],
    ):
    """Visualize ingredients relative frequency per step."""
    recipes = recipes.int()
    num_steps = recipes.shape[0]
    n_ingredients = len(ingredients_name)

    # Calculate relative frequency of each ingredient per step
    ingredients_count = torch.stack([
        torch.bincount(recipes[step], minlength=n_ingredients + 1)
        for step in range(num_steps)
    ])
    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy().T

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        ingredients_proportion,
        aspect='auto',
        cmap='viridis',
        origin='upper'
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Ingredient')
    ax.set_title('Relative frequency of ingredient')
    ax.set_xticks(np.arange(num_steps))
    ax.set_xticklabels(np.arange(num_steps) + 1)
    ax.set_yticks(np.arange(n_ingredients + 1))
    ax.set_yticklabels(ingredients_name + ['Remove'])
    fig.colorbar(im, ax=ax, label='Mean relative frequency')
    fig.tight_layout()
    fig.savefig("../plots/ingredients_heatmap.png")
    plt.show()


def plot_effects_heatmap(
    effects: torch.Tensor,
    effects_name: List[str]
):
    """Visualize the relative frequency of each effect."""
    num_steps = effects.shape[0]
    n_effects = effects.shape[2]

    # Calculate mean presence of each effect per step
    effects_frequency = torch.zeros(
        (num_steps, n_effects),
        dtype=torch.float32
    )
    for step in range(num_steps):
        effects_frequency[step] = effects[step].mean(dim=0).float()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        effects_frequency.cpu().numpy().T,
        aspect='auto',
        cmap='viridis',
        origin='upper'
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Effect')
    ax.set_title('Relative frequency of effects per step')
    ax.set_xticks(np.arange(num_steps))
    ax.set_yticks(np.arange(n_effects))
    ax.set_yticklabels(effects_name)
    fig.colorbar(im, ax=ax, label='Mean relative frequency')
    fig.tight_layout()
    fig.savefig("../plots/effects_heatmap.png")
    plt.show()


def plot_profits_boxplot(profits: torch.Tensor):
    """Visualize profit distribution for each step using box plots."""
    num_steps = profits.shape[0]
    profits = profits.cpu().numpy()  # shape: (num_steps, num_samples)

    steps = np.arange(1, num_steps + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Create a list of arrays, one per step, for boxplot
    data = [profits[step] for step in range(num_steps)]
    ax.boxplot(
        data,
        positions=steps,
        widths=0.6,
        patch_artist=True,
        showfliers=False
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Profit')
    ax.set_title('Profit distribution per step')
    ax.set_xlim(0.5, num_steps + 0.5)
    fig.tight_layout()
    fig.savefig("../plots/profit_boxplot.png")
    plt.show()


def plot_sankey_diagram(
    recipes: torch.Tensor,
    profits: torch.Tensor,
    ingredients_name: list
):
    """Plot a Sankey diagram for ingredient choices at each step.

    Args:
        recipes (torch.Tensor): Tensor of recipes with ingredient ids.
        profits (torch.Tensor): Tensor of profit values.
        ingredients_name (list): List of ingredients name.
    """
    num_steps, num_simulations = recipes.shape
    n_ingredients = len(ingredients_name)
    recipes = recipes.int()

    # Build node labels
    node_labels = []
    node_map = {}
    ingridients_id = np.arange(n_ingredients)
    for step in range(num_steps):
        for ingredient in ingridients_id:
            if ingredient < n_ingredients:
                label = ingredients_name[ingredient]
                node_map[(step, ingredient)] = len(node_labels)
                node_labels.append(label)

    # Accumulate links weighted by profit at target step
    link_values = {}
    for sim in range(num_simulations):
        for step in range(num_steps - 1):
            # Current and next ingredient
            # (step, ingredient) -> (step + 1, next_ingredient)
            current_ingredient = recipes[step, sim].item()
            next_ingredient = recipes[step + 1, sim].item()
            if (
                next_ingredient < n_ingredients and
                current_ingredient < n_ingredients
            ):
                # Get source and target node indices
                source = node_map[(step, current_ingredient)]
                target = node_map[(step + 1, next_ingredient)]

                # Add link value
                value = float(profits[step + 1, sim])
                link_values[(source, target)] = link_values.get(
                    (source, target),
                    0
                ) + value

    sources, targets, values = zip(
        *[
            (s, t, v) for (s, t), v in link_values.items()
        ]
    )
    values = np.array(values)
    values_scaled = values / values.max()
    colors = [
        f'rgba({int(255 * (1 - v))},'
        f'{int(255 * (1 - v))},'
        f'{int(255 * (1 - v))}, 0.5)'
        for v in values_scaled
    ]

    # Plot
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
        )
    )
    fig.update_layout(title_text="Recipe Profit Sankey Diagram", font_size=12)
    fig.savefig("../plots/recipes_sankey.png")
    fig.show()
