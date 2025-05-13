"""Script to create Mixing Chain."""
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm


class MixingChain:
    """Class to simulate the mixing chain."""
    def __init__(self):
        """Mixing chain setup."""
        # Load data
        self.ingredients_df = pd.read_json("data/ingredients.json")
        self.products_df = pd.read_json("data/products.json")
        self.rules_df = pd.read_json("data/rules.json")
        self.effects_df = pd.read_json("data/effects.json")

        # Create fast access dictionaries
        self.ingredient_cost = dict(
            zip(self.ingredients_df['name'], self.ingredients_df['cost'])
        )
        self.effect_multiplier = dict(
            zip(self.effects_df['name'], self.effects_df['multiplier'])
        )
        self.ingredient_effect = dict(
            zip(self.ingredients_df['name'], self.ingredients_df['effect'])
        )

        # Create rules dictionary
        self.rules_dict = {}
        for _, row in self.rules_df.iterrows():
            self.rules_dict[
                (row['ingredient'], row['effect_base'])
            ] = row['effect_result']

        # Create probability for each ingredient
        total_inverse_cost = (1 / self.ingredients_df["cost"]).sum()
        self.ingredients_df["prob"] = (
            1 / self.ingredients_df["cost"]
        ) / total_inverse_cost

        # Create effect and ingredient sets
        self.all_effects = set(self.effects_df['name'])
        self.all_ingredients = set(self.ingredients_df['name'])

    def _cost(self, state: dict) -> float:
        """Optimized cost calculation."""
        ingredients_cost = sum(
            self.ingredient_cost[ing] for ing in state["ingredients"]
        )
        return float(self.base_product["cost"].iloc[0] + ingredients_cost)

    def _value(self, state: dict) -> float:
        """Optimized value calculation."""
        mult = sum(self.effect_multiplier[eff] for eff in state["effects"])
        return float((1 + mult) * self.base_product["value"].iloc[0])

    def _apply_ingedient(self, state: dict, ingredient: pd.Series) -> dict:
        """Optimized ingredient application."""
        new_state = copy.deepcopy(state)
        new_state["ingredients"].append(ingredient["name"])
        current_effects = set(new_state["effects"])
        new_effect = ingredient["effect"]

        # Apply base effect if not already present and there's space
        if new_effect not in current_effects and len(current_effects) < 8:
            new_state["effects"].append(new_effect)
            current_effects.add(new_effect)

        # Apply rules to all current effects (including the new one if added)
        effects_to_process = list(new_state["effects"])  # Create copy for iteration
        for effect in effects_to_process:
            key = (ingredient["name"], effect)
            if key in self.rules_dict:
                resulting_effect = self.rules_dict[key]
                # Only apply if the resulting effect isn't already present
                if resulting_effect not in current_effects:
                    new_state["effects"].remove(effect)
                    new_state["effects"].append(resulting_effect)
                    current_effects.remove(effect)
                    current_effects.add(resulting_effect)
                else:
                    # If resulting effect exists, just remove the original
                    new_state["effects"].remove(effect)
                    current_effects.remove(effect)

        return new_state

    def _find_neighbours(self, state: dict) -> Tuple[List[dict], np.ndarray]:
        """Vectorized neighbour finding."""
        neighbour_states = [
            self._apply_ingedient(state, self.ingredients_df.iloc[i])
            for i in range(len(self.ingredients_df))
        ]

        # Acceptance parameter
        current_diff = abs(self._value(state) - self._cost(state))
        neighbour_diffs = np.array(
            [abs(self._value(s) - self._cost(s))
            for s in neighbour_states]
        )
        acceptances = np.minimum(1, neighbour_diffs / current_diff)

        # Calculate probabilities
        probs = self.ingredients_df["prob"].to_numpy() * acceptances
        probs /= probs.sum()  # Normalize

        return neighbour_states, probs

    def simulate(
        self,
        base_product: str,
        num_steps: int
    ) -> Tuple[List[dict], List[float], List[float]]:
        """Simulate chain starting with base product.

        Args:
            base_product (str): Base product used as source state.
            num_steps (int): Number of simulation steps.

        Raises:
            ValueError: If chosen based product.
        """        # Set base state
        self.base_product = self.products_df[
            self.products_df["name"] == base_product
        ]
        if len(self.base_product) == 0:
            raise ValueError(f"Base product {base_product} is invalid!")

        state = {
            "ingredients": [],
            "effects": [self.base_product["effect"].iloc[0]],
        }

        states = []
        costs = []
        values = []

        for _ in tqdm(range(num_steps), desc="Simulating steps"):
            neighbours_states, neighbours_probs = self._find_neighbours(state)
            state = np.random.choice(a=neighbours_states, p=neighbours_probs)

            states.append(copy.deepcopy(state))
            costs.append(self._cost(state))
            values.append(self._value(state))

        return states, np.array(costs), np.array(values)


chain = MixingChain()
states, costs, values = chain.simulate("OG Kush", 1000)
profits = values - costs
idx_max = np.where(profits == profits.max())[0][0]
print(f"Idx: {idx_max}.\nIngredientes:{states[idx_max]["ingredients"]}.\nEfeitos:{states[idx_max]["effects"]}\nCusto: {costs[idx_max]}.\nValor: {values[idx_max]}.\nProfit: {profits[idx_max]}")