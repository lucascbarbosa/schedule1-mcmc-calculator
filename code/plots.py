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


def visualize_profits(
    profits: torch.Tensor):
    """Visualize profit time series."""
    num_steps = profits.shape[0]
    profits = profits.cpu().numpy()

    steps = np.arange(1, num_steps + 1)

    mean = profits.mean(axis=1)
    sem = profits.std(axis=1) / np.sqrt(profits.shape[1])
    ci_upper = mean + 1.96 * sem
    ci_lower = mean - 1.96 * sem

    # Plot profits
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        mean,
        linewidth=2,
    )
    plt.fill_between(
        steps,
        ci_lower,
        ci_upper,
        alpha=0.3,
    )
    plt.xlabel('Step')
    plt.ylabel('Profit')
    plt.title('Profit per step')
    plt.grid(True)
    fig.tight_layout()
    fig.savefig("../plots/profits_series.png")


def visualize_effects(
    effects: torch.Tensor):
    """Visualize effects time series."""
    num_steps = effects.shape[0]
    effects = effects.cpu().numpy()

    steps = np.arange(1, num_steps + 1)

    sum = effects.sum(axis=1)
    mean = sum.mean(axis=1)
    sem = sum.std(axis=1) / np.sqrt(sum.shape[1])
    ci_upper = mean + 1.96 * sem
    ci_lower = mean - 1.96 * sem

    # Plot effects
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        mean,
        linewidth=2,
    )
    plt.fill_between(
        steps,
        ci_lower,
        ci_upper,
        alpha=0.3,
    )
    plt.xlabel('Step')
    plt.ylabel('Number of active effects')
    plt.title('Number of active effects per step')
    plt.grid(True)
    fig.tight_layout()
    fig.savefig("../plots/effects_series.png")
    plt.show()
