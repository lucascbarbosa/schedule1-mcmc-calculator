"""Script to create Mixing Chain."""
import concurrent.futures
import copy
import numpy as np
import pandas as pd
from arrays import DatabaseArrays
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Tuple


class ChainState:
    """Arrays related to chain state.

    The ingredients array maps the ingredients used in the recipe, i.e.,
    the path traveled in the chain.

    The effects array indicates the active effects, i.e., the state of
    the chain.

    """
    def __init__(
            self,
            chain,
            base_product: str,
    ):
        """Create ingredients and effects arrays.

        Args:
            base_product (str): Base product used as source state.

        Raises:
            ValueError: If chosen based product.
        """
        # Get base product cost, value and effect
        base_product = chain.products_df[
            chain.products_df["product_name"] == base_product]
        if len(base_product) == 0:
            raise ValueError(f"Base product {base_product} is invalid!")
        self.product_value = base_product["value"].iloc[0]

        # Create an integer ingredients array.
        # Each element of the array represents how many times that
        # ingredient was used.
        self.ingredients_count = np.zeros((chain.n_ingredients, 1))

        # Create a sparse binary effects array.
        # Earch element of the array indicates if effect is present.
        self.active_effects = np.zeros((chain.n_effects, 1))
        self.active_effects[
            chain.effects_df["effect_name"] ==
            base_product["effect_name"].iloc[0]
        ] = 1

    def cost(self) -> float:
        """Calculates the cost of product state."""
        return float((
            chain.ingredients_cost @ self.ingredients_count
        ).item())

    def value(self) -> float:
        """Calculates sell value of product state."""
        mult = float((chain.effects_multiplier @ self.active_effects).item())
        return (1 + mult) * float(self.product_value)

    def mix_ingredient(self, ingredient_id: int):
        """Mix product with ingredient."""
        #  Increment ingredient count
        self.ingredients_count[ingredient_id] += 1

        # Apply effects transition rules
        self.active_effects = (
            (chain.rules[ingredient_id, :, :] @ self.active_effects) > 0
        ).astype(int)

        # Add ingredient effect
        if self.active_effects.sum() < 8:
            self.active_effects[chain.ingredients_effect[0, ingredient_id]] = 1

    def compute_ingredient_prob(self) -> Tuple[Iterable[dict], np.ndarray]:
        """Compute adjusted probability of using each ingredient.

        Given a current state, the neighbour states are the results
        from adding an ingredient to the product.
        """
        # Generate neighbours ingredients arrays
        neighbours_ingredients = (
            np.eye(chain.n_ingredients) + self.ingredients_count
        )

        # Generate neighbours effects arrays
        neighbours_effects = (
            (chain.rules @ self.active_effects) > 0
        ).astype(int).reshape((chain.n_ingredients, chain.n_effects)).T

        # Add ingredients effects
        for i in range(chain.n_ingredients):
            if neighbours_effects[:, i].sum() < 8:
                neighbours_effects[chain.ingredients_effect[0, i], i] = 1

        # Acceptance parameter
        neighbours_values = (
            (1 + (chain.effects_multiplier @ neighbours_effects))
            * self.product_value
        )
        neighbours_costs = (
                chain.ingredients_cost @ neighbours_ingredients
        )
        neighbours_profit = (neighbours_values - neighbours_costs).ravel()
        current_profit = self.value() - self.cost()
        acceptances = np.clip(neighbours_profit / current_profit, 0.0, 1.0)

        # Calculate probabilities
        probs = acceptances / acceptances.sum()  # Normalize
        return probs.ravel()


class ChainSimulation:
    """Class to simulate the mixing chain."""
    def __init__(self):
        """Mixing chain setup."""
        # Load data
        current_dir = Path(__file__).parent
        ingredients_df = pd.read_json(
            current_dir.parent / "data/ingredients.json")
        rules_df = pd.read_json(
            current_dir.parent / "data/rules.json")
        effects_df = pd.read_json(
            current_dir.parent / "data/effects.json")
        products_df = pd.read_json(
            current_dir.parent / "data/products.json")

        # Dimensions
        self.n_ingredients = len(ingredients_df)
        self.n_effects = len(effects_df)

        ##########################
        # Create arrays database #
        # Convert from name to index
        (
            self.ingredients_df,
            self.effects_df,
            self.products_df,
            self.rules_df,
        ) = DatabaseArrays.convert_name_to_index(
            ingredients_df=ingredients_df,
            effects_df=effects_df,
            products_df=products_df,
            rules_df=rules_df
        )
        # Fetch ingredients arrays
        (
            self.ingredients_effect,  # Array with ingredients effects
            self.ingredients_cost,  # Array with ingredients cost
        ) = DatabaseArrays.create_ingredients_arrays(
            ingredients_df=self.ingredients_df)

        # Fetch effects arrays
        (
            self.effects_multiplier  # Array with effects multiplier
        ) = DatabaseArrays.create_effects_arrays(effects_df=self.effects_df)

        # Fetch rules arrays
        self.rules = DatabaseArrays.create_effects_rules_matrix(
            effects_df=self.effects_df,
            ingredients_df=self.ingredients_df,
            rules_df=self.rules_df
        )  # Array with effects transition rules

    def _simulate(
        self, args
    ) -> Tuple[Iterable[list], np.ndarray, np.ndarray, np.ndarray]:
        """Simulate chain starting with base product."""
        base_product, num_steps = args
        # Set base state
        state = ChainState(
            chain=self,
            base_product=base_product,
        )
        recipe = np.zeros(num_steps)
        for i in range(num_steps):
            # Choose ingredient based on metropoles-hastings adjusted prob
            ingredients_prob = state.compute_ingredient_prob()
            ingredient = np.random.choice(
                a=range(self.n_ingredients), p=ingredients_prob)

            # Mix ingredient
            state.mix_ingredient(ingredient)

            # Store ingredient in recipe
            recipe[i] = int(ingredient)

        return recipe, state.active_effects, state.cost(), state.value()

    def parallel_simulation(
        self,
        base_product: str,
        num_simulations: int,
        num_steps: int = 8,
    ) -> Tuple[Iterable[list], np.ndarray, np.ndarray, np.ndarray]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            num_simulations (int): Number of parallel simulations.
            num_steps (int, optional): Max number of simulation steps. Defaults to 8.

        Returns:
            Tuple[Iterable[list], np.ndarray, np.ndarray, np.ndarray]: Results
            from the simulation.
        """
        args_list = [(base_product, num_steps) for _ in range(num_simulations)]

        # Run simulations
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(
                executor.map(self._simulate, args_list), total=num_simulations
            ):
                results.append(result)

        # Merge results
        (
            all_recipes,
            all_effects,
            all_costs,
            all_values,
        ) = zip(*results)

        # Process results
        all_recipes = np.array(all_recipes).reshape((
            num_simulations, num_steps))
        all_effects = np.array(all_effects).reshape((
            num_simulations, self.n_effects))
        all_costs = np.array(all_costs).reshape(num_simulations)
        all_values = np.array(all_values).reshape(num_simulations)
        all_profits = all_values - all_costs
        return all_recipes, all_effects, all_costs, all_values, all_profits

    def decode_effects(self, effects: np.ndarray):
        """Convert effects from id to name."""
        effects_id = np.where(effects == 1)[0]
        return self.effects_df[
            self.effects_df["effect_id"].isin(effects_id)]["effect_name"].tolist()

    def decode_recipe(self, recipe: list):
        """Convert ingredients from id to name."""
        recipe_converted = []
        for ingredient in recipe:
            recipe_converted.append(
                self.ingredients_df[
                    self.ingredients_df["ingredient_id"] == ingredient
                ]["ingredient_name"].iloc[0]
            )
        return recipe_converted


chain = ChainSimulation()
recipes, effects, costs, values, profits = chain.parallel_simulation(
    "Cocaine", num_simulations=1_000_000, num_steps=8)
id_max = np.where(profits == profits.max())[0][0]
print(f"\n\nOTIMIZADO:.\nReceitas:{chain.decode_recipe(recipes[id_max])}.\nEfeitos:{chain.decode_effects(effects[id_max])}\nCusto: {costs[id_max]}.\nValor: {values[id_max]}.\nProfit: {profits[id_max]}")
