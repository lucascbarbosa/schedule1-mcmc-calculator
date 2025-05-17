"""Script to visualize plots of optimization."""
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_ingredients_choice(
    recipes: torch.Tensor,
    n_ingredients: int):
    """Visualize ingredients choice in bar plots."""
    num_steps = recipes.shape[0]
    ingredients_count = torch.stack([
        torch.bincount(recipes[step], minlength=n_ingredients)
        for step in range(num_steps)
    ])
    ingredients_proportion = (
        ingredients_count /
        ingredients_count.sum(dim=1, keepdim=True)
    ).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    steps = np.arange(1, num_steps + 1)
    bottom = np.zeros(num_steps)
    cmap = (
        plt.get_cmap('tab20')
        if n_ingredients <= 20
        else plt.cm.get_cmap('hsv', n_ingredients)
    )
    colors = [cmap(i) for i in range(n_ingredients)]

    for i in range(n_ingredients):
        ax.bar(
            steps,
            ingredients_proportion[:, i],
            bottom=bottom,
            width=0.8,
            color=colors[i],
            label=f'Ingredient {i + 1}'
        )
        bottom += ingredients_proportion[:, i]

    # Configurações do gráfico
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Ingredient proportion')
    ax.set_title('Ingredient choice proportion per simulation step')
    ax.set_xticks(steps)
    ax.set_yticks(torch.linspace(0, 1, 11))
    ax.set_yticklabels([f'{int(y * 100)}%' for y in torch.linspace(0, 1, 11)])
    ax.legend(title='Ingredients', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig("../plots/ingredients_choice.png")


def visualize_profits(profits: torch.Tensor):
    """Visualize profit time series."""
    num_steps = profits.shape[0]
    batch_size = profits.shape[1]

    profits = profits.cpu().numpy()
    means = profits.mean(axis=1)
    sems = profits.std(axis=1) / np.sqrt(batch_size)
    ci_upper = means + 1.96 * sems
    ci_lower = means - 1.96 * sems
    steps = np.arange(1, num_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, means, linewidth=2)
    ax.fill_between(
        steps,
        ci_lower,
        ci_upper,
        color='tab:blue',
        alpha=0.3,
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('Profit')
    ax.set_title('Profit mean and confidence interval per step')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("../plots/profits_series.png")
    plt.show()
