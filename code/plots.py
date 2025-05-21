"""Script to visualize plots of optimization."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def visualize_ingredients_choice(
    ingredients: torch.Tensor,
    ingredients_df: pd.DataFrame,
    n_ingredients: int):
    """Visualize ingredients choice in bar plots."""
    ingredients = ingredients.int()
    num_steps = ingredients.shape[0]
    ingredients_count = torch.stack([
        torch.bincount(ingredients[step], minlength=n_ingredients + 1)
        for step in range(num_steps)
    ])
    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy()  # shape: (num_steps, n_ingredients+1)

    steps = np.arange(1, num_steps + 1)
    # Cada linha: ingrediente, cada coluna: step
    data = ingredients_proportion.T  # shape: (n_ingredients+1, num_steps)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = (
        plt.get_cmap('tab20')
        if n_ingredients <= 20
        else plt.cm.get_cmap('hsv', n_ingredients)
    )
    colors = [cmap(i) for i in range(n_ingredients + 1)]
    labels = [ingredients_df.iloc[i]['ingredient_name'] if i < n_ingredients else 'REMOVE' for i in range(n_ingredients + 1)]

    ax.stackplot(steps, data, labels=labels, colors=colors)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Ingredient proportion')
    ax.set_title('Ingredient choice proportion per simulation step')
    ax.set_xlim(1, num_steps)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f'{int(y * 100)}%' for y in np.linspace(0, 1, 11)])
    ax.legend(title='Ingredients', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig("../plots/ingredients_choice.png")


def visualize_profits_effects(
        profits: torch.Tensor,
        effects: torch.Tensor):
    """Visualize profit and effects time series."""
    num_steps = profits.shape[0]

    profits = profits.cpu().numpy()
    effects = effects.cpu().numpy()

    # Fetch mean and confiudence interval of profits and effects
    steps = np.arange(1, num_steps + 1)

    effects_sum = effects.sum(axis=1)
    effects_mean = effects_sum.mean(axis=1)
    effects_sem = effects_sum.std(axis=1) / np.sqrt(effects_sum.shape[1])
    effects_ci_upper = effects_mean + 1.96 * effects_sem
    effects_ci_lower = effects_mean - 1.96 * effects_sem

    profits_mean = profits.mean(axis=1)
    profits_sem = profits.std(axis=1) / np.sqrt(profits.shape[1])
    profits_ci_upper = profits_mean + 1.96 * profits_sem
    profits_ci_lower = profits_mean - 1.96 * profits_sem

    # Plot profits and effects
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        profits_mean,
        linewidth=2,
        label='Profit'
    )
    plt.fill_between(
        steps,
        profits_ci_lower,
        profits_ci_upper,
        alpha=0.3,
    )
    plt.plot(
        steps,
        effects_mean,
        linewidth=2,
        label='Number of active effects'
    )
    plt.fill_between(
        steps,
        effects_ci_lower,
        effects_ci_upper,
        alpha=0.3,
    )
    plt.xlabel('Step')
    plt.ylabel('Profit')
    plt.title('Profit and number of active effects per step')
    plt.grid(True)
    fig.tight_layout()
    fig.savefig("../plots/profits_effects_series.png")
    plt.legend()
    plt.show()
