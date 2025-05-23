"""Script to plot optimization results."""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from typing import List


def plot_final_step_ingredients_barplot(
    recipes: torch.Tensor,
    ingredients_name: List[str],
    recipe_size: int = 7,
    save_path: str = "../plots/final_step_stacked_bar.svg"
):
    """Plot ingredients relative frequency per recipe position in final step."""
    # Get last step: shape (recipe_size, batch_size)
    last_step = recipes[-1]  # shape: (recipe_size, batch_size)
    n_ingredients = len(ingredients_name)
    batch_size = last_step.shape[1]

    # For each position in the recipe, count ingredient occurrences
    counts = torch.zeros((recipe_size, n_ingredients), dtype=torch.float32)
    for pos in range(recipe_size):
        pos_ids = last_step[pos].to(torch.int64)
        counts[pos] = torch.bincount(pos_ids, minlength=n_ingredients)

    # Convert to relative frequency
    rel_freq = (counts / counts.sum(dim=1, keepdim=True)).cpu().numpy()  # shape: (recipe_size, n_ingredients)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    indices = np.arange(1, recipe_size + 1)
    bottom = np.zeros(recipe_size)
    for i, name in enumerate(ingredients_name):
        ax.bar(
            indices,
            rel_freq[:, i],
            bottom=bottom,
            label=name
        )
        bottom += rel_freq[:, i]

    ax.set_xlabel("Recipe Position")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(
        "Ingredient relative frequency per recipe position (at last step)")
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices])
    ax.legend(title="Ingredient", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


def plot_ingredients_lineplot(
    recipes: torch.Tensor,
    ingredients_name: List[str],
):
    """Plot ingredients relative frequency per step as line plot."""
    recipes = recipes.int()
    n_steps = recipes.shape[0]
    n_ingredients = len(ingredients_name)

    # Flatten all batches for each step and count ingredient occurrences
    ingredients_count = torch.stack([
        torch.bincount(
            recipes[step].reshape(-1), minlength=n_ingredients
        )
        for step in range(n_steps)
    ])
    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy().T  # shape: (n_ingredients, n_steps)

    steps = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(ingredients_name):
        ax.plot(steps, ingredients_proportion[i], label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative frequency')
    ax.set_title('Relative frequency of ingredients per simulation step')
    ax.set_xticks(steps[::10])
    ax.legend(title='Ingredient')
    fig.tight_layout()
    fig.savefig("../plots/ingredients_lineplot.svg")
    plt.show()


def plot_effects_lineplot(
    effects: torch.Tensor,
    effects_name: List[str]
):
    """Plot the relative frequency of each effect as a line plot."""
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

    steps = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(effects_name):
        ax.plot(steps, effects_frequency[i], label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative frequency')
    ax.set_title('Relative frequency of effects per step')
    ax.set_xticks(steps[::10])
    ax.legend(title='Effect')
    # fig.tight_layout()
    fig.savefig("../plots/effects_lineplot.svg")
    plt.show()


def plot_profit_lineplot(profits: torch.Tensor):
    """Plot mean profit per step with 95% confidence interval as a line plot."""
    n_steps = profits.shape[0]
    profits_np = profits.cpu().numpy()  # shape: (n_steps, n_samples)

    steps = np.arange(1, n_steps + 1)
    means = profits_np.mean(axis=1)
    stds = profits_np.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, means, label='Mean Profit', color='blue')
    ax.fill_between(
        steps, means - stds, means + stds, color='blue', alpha=0.2
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Profit')
    ax.set_title('Mean profit per step with 95% confidence interval')
    ax.set_xlim(0.5, n_steps + 0.5)
    fig.tight_layout()
    fig.savefig("../plots/profit_lineplot.svg")
    plt.show()


def plot_recipes_sankey(
    recipes: torch.Tensor,
    profits: torch.Tensor,
    ingredients_name: list
):
    """Plot a Sankey diagram for recipe at each step.

    Args:
        recipes (torch.Tensor): Tensor of recipes with ingredient ids.
        profits (torch.Tensor): Tensor of profit values.
        ingredients_name (list): List of ingredients name.
    """
    # Fetch dimensions
    n_steps, n_batches = recipes.shape
    n_ingredients = len(ingredients_name)

    # Build node labels
    node_labels = []
    node_map = {}
    ingridients_id = np.arange(n_ingredients)
    for step in range(n_steps):
        for ingredient in ingridients_id:
            if ingredient < n_ingredients:
                label = ingredients_name[ingredient]
                node_map[(step, ingredient)] = len(node_labels)
                node_labels.append(label)

    # Accumulate links weighted by profit at target step
    link_values = {}
    for sim in range(n_batches):
        for step in range(n_steps - 1):
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
                value = float(
                    profits[step + 1, sim] / profits.max(dim=1)[0][step + 1]
                )
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
        ),
    )
    fig.update_layout(
        width=1200,
        height=600,
        title_text="Recipe Profit Sankey Diagram",
        font_size=12
    )
    fig.write_image("../plots/recipe_sankey.svg")
    fig.show()
