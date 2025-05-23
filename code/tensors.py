"""Script to create tensors used in simulation."""
import pandas as pd
import torch
from pathlib import Path
from typing import Union


class Database:
    """Class to create tensors from dataframes."""
    def __init__(self):
        """Convert ingredients, effects and products names to id."""
        # Load data
        dir = Path(__file__).parent
        ingredients_df = pd.read_json(
            dir.parent / "data/ingredients.json")
        rules_df = pd.read_json(
            dir.parent / "data/rules.json")
        effects_df = pd.read_json(
            dir.parent / "data/effects.json")
        products_df = pd.read_json(
            dir.parent / "data/products.json")

        ingredient_to_id = ingredients_df.reset_index().set_index(
            "name")["index"]
        effect_to_id = effects_df.reset_index().set_index(
            "name")["index"]
        product_to_id = products_df.reset_index().set_index(
            "name")["index"]

        # Ingredients
        ingredients_df["ingredient_id"] = ingredients_df["name"].map(
            ingredient_to_id)
        ingredients_df["effect_id"] = ingredients_df["effect"].map(
            effect_to_id)
        ingredients_df = ingredients_df.rename(
            columns={"name": "ingredient_name", "effect": "effect_name"})
        self.ingredients_df = ingredients_df
        self.n_ingredients = len(ingredients_df)

        # Effects
        effects_df["effect_id"] = effects_df["name"].map(effect_to_id)
        effects_df = effects_df.rename(columns={"name": "effect_name"})
        self.n_effects = len(effects_df)
        self.effects_df = effects_df

        # Products
        products_df["product_id"] = products_df["name"].map(product_to_id)
        products_df["effect_id"] = products_df["effect"].map(effect_to_id)
        products_df = products_df.rename(
            columns={"name": "product_name", "effect": "effect_name"})
        self.products_df = products_df

        # Rules
        rules_df["ingredient"] = rules_df[
            "ingredient"].map(ingredient_to_id)
        rules_df["effect_base"] = rules_df[
            "effect_base"].map(effect_to_id)
        rules_df["effect_result"] = rules_df[
            "effect_result"].map(effect_to_id)
        self.rules_df = rules_df

        # Set torch device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fetch ingredients arrays
        self.create_ingredients_tensors()

        # Fetch effects multiplier tensors
        self.create_effects_tensors()

        # Fetch rules tensor
        self.create_effects_rules_tensor()

    def create_ingredients_tensors(self):
        """Create ingredients tensors."""
        self.ingredients_effect = torch.from_numpy(
            self.ingredients_df['effect_id'].to_numpy().reshape((1, -1))
        ).to(dtype=torch.float32, device=self.device)

        self.ingredients_cost = torch.from_numpy(
            self.ingredients_df['cost'].to_numpy().reshape((1, -1))
        ).to(dtype=torch.float32, device=self.device)

    def create_effects_tensors(self):
        """Create effects tensors."""
        self.effects_multiplier = torch.from_numpy(
            self.effects_df['multiplier'].to_numpy().reshape((1, -1)),
        ).to(dtype=torch.float32, device=self.device)

    def create_effects_rules_tensor(self) -> torch.Tensor:
        """Create rules of mixing effects tensor.

        This is also a sparse binary tensor of shape (n_ingredients x n_effects
        x n_effects) indicating the effects conversion for each ingredient.

        """
        self.rules = torch.eye(
            self.n_effects, dtype=torch.float32, device=self.device
        ).repeat(self.n_ingredients, 1, 1)
        for _, row in self.rules_df.iterrows():
            self.rules[
                row["ingredient"],
                row["effect_result"],
                row["effect_base"]
            ] = 1
            self.rules[
                row["ingredient"],
                row["effect_base"],
                row["effect_base"]
            ] = 0


class State(Database):
    """Tensors related to chain state for batched simulation.

    The ingredients array maps the ingredients used in the recipe, i.e.,
    the path traveled in the chain.

    The effects array indicates the active effects, i.e., the state of
    the chain.

    """
    def __init__(
        self,
        base_product: str,
        batch_size: int,
        recipe_size: int,
    ):
        """Create initial ingredients and effects tensors.

        Args:
            base_product (str): Base product used as source state.
            batch_size (int): Number of simulations in batch.
            recipe_size (int): Number of steps in simulation.

        Raises:
            ValueError: If chosen based product.
        """
        # Define global variables
        super().__init__()
        self.batch_size = batch_size
        self.recipe_size = recipe_size

        # Get base product cost, value and effect
        base_product = self.products_df[
            self.products_df["product_name"] == base_product]
        if len(base_product) == 0:
            raise ValueError(f"Base product {base_product} is invalid!")
        self.product_effect = (
            self.effects_df["effect_name"] ==
            base_product["effect_name"].iloc[0]
        )
        self.product_value = base_product["value"].iloc[0]

        # Create state tensors
        self.recipes = self.create_recipes()
        self.effects = self.mix_recipes(self.recipes)

    def create_recipes(self) -> torch.Tensor:
        """Create a random recipe tensor.

        Tensor of shape (recipe_size, batch_size) storing the ingredient id
        used in a given recipe simulation and recipe step.
        """
        return torch.randint(
            0, self.n_ingredients,
            (self.recipe_size, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )

    def create_effects(self) -> torch.Tensor:
        """Create a sparse binary effects tensor.

        Tensor of shape (recipe_size, batch_size, n_effects) indicating
        if the effect at a given recipe step and batch simulation is active.
        """
        effects = torch.zeros(
            (self.n_effects, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )

        # Define initial effect as the effect of the base product
        effects[self.product_effect, :] = 1

        return effects

    def get_recipes(self) -> torch.Tensor:
        """Get current state (recipes) tensor."""
        return self.recipes.clone()

    def get_neighbour_recipes(self) -> torch.Tensor:
        """Get neighbour recipes tensor."""
        return self.neighbour_recipes.clone()

    def get_effects(self) -> torch.Tensor:
        """Get current active effects tensor."""
        return self.effects.clone()

    def get_neighbour_effects(self) -> torch.Tensor:
        """Get neighbour active effects tensor."""
        return self.neighbour_effects.clone()

    def ingredients_count(self, recipes: torch.Tensor) -> torch.Tensor:
        """Count each ingredient in recipes for each batch (column).

        Returns a tensor of shape (n_ingredients, batch_size) where each column
        contains the count of each ingredient for that batch.
        """
        one_hot = torch.nn.functional.one_hot(
            recipes.long(), num_classes=self.n_ingredients
        )
        ingredients_count = one_hot.sum(dim=0).T.float()
        return ingredients_count

    def cost(self, recipes: Union[torch.Tensor, None] = None) -> float:
        """Calculates the cost of the recipe."""
        if recipes is None:
            recipes = self.get_recipes()
        ingredients_count = self.ingredients_count(recipes)
        return (
            self.ingredients_cost @ ingredients_count
        )

    def value(self, effects: Union[torch.Tensor, None] = None) -> float:
        """Calculates sell value of the resulting product."""
        if effects is None:
            effects = self.get_effects()
        mult = self.effects_multiplier @ effects
        return (1 + mult) * float(self.product_value)

    def profit(self) -> float:
        """Calculates profit of current state."""
        effects = self.get_effects()
        recipes = self.get_recipes()
        return self.value(effects) - self.cost(recipes)

    def neighbour_profit(self) -> float:
        """Calculates profit of neighbour state."""
        effects = self.get_neighbour_effects()
        recipes = self.get_neighbour_recipes()
        return self.value(effects) - self.cost(recipes)

    def effects_distance(self, desired_effects: torch.Tensor) -> float:
        """Distance current state active and desired effects."""
        return (
            -torch.sum((self.effects - desired_effects)**2, dim=0)
        )

    def apply_ingredients_effect(
        self,
        ingredients: torch.Tensor,
        effects: torch.Tensor,
    ) -> torch.Tensor:
        """Apply effect from added ingredients."""
        # Check if sum of effects is leq 8
        sum_effects = effects.sum(dim=0)
        sum_mask = sum_effects < 8

        # Apply ingredient effects only if sum of effects is leq 8
        effect_ids = self.ingredients_effect[0, ingredients]
        filtered_batch_ids = torch.nonzero(sum_mask).squeeze(1)
        filtered_effect_ids = effect_ids[sum_mask]
        effects[filtered_effect_ids.to(int), filtered_batch_ids] = 1

        return effects

    def apply_effects_rules(
        self,
        ingredients: torch.Tensor,
        effects: torch.Tensor,
    ) -> torch.Tensor:
        """Apply effects transition rules for each ingredient."""
        # Fetch rules from each added ingredient
        rules_batch = self.rules[ingredients]
        # shape: (batch_size, n_effects, n_effects)

        # Compute rules
        effects_result = torch.bmm(
            rules_batch,
            effects.T.unsqueeze(2)
        ).squeeze(2).T  # shape: (n_effects, batch_size)

        # Detect all duplicates effects (> 1.0) and clamp them back to 1.0
        duplicated_mask = effects_result > 1.0
        effects_result = effects_result.clamp_max(1.0)

        # Restore wrongly deactivated effects
        if duplicated_mask.any():
            # Get duplicated effects
            effect_ids, ingredient_ids = duplicated_mask.nonzero(as_tuple=True)
            active_sources = (
                rules_batch.bool() & effects.T.unsqueeze(2).bool()
            ).float()
            active_sources -= torch.eye(
                self.n_effects, device=self.device
            ).repeat(effects_result.shape[1], 1, 1).reshape(
                (effects_result.shape[1], self.n_effects, self.n_effects)
            )
            source_effect_ids = torch.argmax(active_sources.int(), dim=2)
            effects_result[
                source_effect_ids[ingredient_ids, effect_ids],
                ingredient_ids
            ] = 1.0

        return effects_result

    def mix_ingredients(
        self,
        recipe_step: torch.Tensor,
        effects: torch.Tensor
    ) -> torch.Tensor:
        """Mix products with ingredients in batch."""
        # Apply effects transition rules
        effects_result = self.apply_effects_rules(recipe_step, effects)

        # Apply ingredient effect
        effects_result = self.apply_ingredients_effect(
            recipe_step, effects_result)

        return effects_result

    def create_neighbour(self) -> torch.Tensor:
        """Create neigbour recipes and effects tensors."""
        neighbour_recipes = self.get_recipes()

        # Randomly choose ingredients to change id
        change_ids = torch.randint(
            0, neighbour_recipes.shape[0],
            (neighbour_recipes.shape[1],),
            dtype=torch.int,
            device=self.device
        )
        current_values = neighbour_recipes[
            change_ids,
            torch.arange(change_ids.shape[0])
        ]

        # Randomly generate new ingredient ids (different from the current one)
        change_values = torch.randint(
            0, self.n_ingredients - 1,
            (neighbour_recipes.shape[1],),
            dtype=torch.float32,
            device=self.device
        )

        # If new value >= current, increment by 1 to skip current value
        mask = (change_values >= current_values)
        change_values[mask] += 1

        # Ensure new values do not exceed self.n_ingredients - 1
        change_values = change_values.clamp(max=self.n_ingredients - 1)
        neighbour_recipes[
            change_ids,
            torch.arange(change_ids.shape[0])
        ] = change_values
        self.neighbour_recipes = neighbour_recipes

        # Compute neighbour effects from recipe
        neighbour_effects = self.mix_recipes(neighbour_recipes)
        self.neighbour_effects = neighbour_effects

    def neighbours_acceptance(self, temperature: torch.Tensor) -> torch.Tensor:
        """Calculates probability of choosing each ingredient.

        This is a Boltzmann probability adjusted by the Metropoles-Hastings
        acceptance parameter that takes into account the profit resulting from
        adding that ingredient. The temperature parameter is computed via
        simulated annealing with log schedule.

        Args:
            temperature (torch.Tensor): Current Boltzmann temperature.

        Returns:
            torch.Tensor: neighbour acceptance
        """
        # Compute objective function for current and neighbour states
        current_obj = self.profit()
        neighbour_obj = self.neighbour_profit()

        # Compute acceptance probability
        acceps = torch.clamp(
            torch.exp((neighbour_obj - current_obj) / temperature),
            max=1.0
        )

        # Decide if neighbour will be accepted
        accept_neighbour = torch.rand_like(acceps) <= acceps
        return accept_neighbour

    def walk(self, temperature: torch.Tensor):
        """Compute next step in the Markov Chain random walk.

        Args:
            temperature (torch.Tensor): Current Boltzmann temperature.
        """
        # Create neighbour recipes and effects
        self.create_neighbour()

        # Compute acceptances for each neighbour
        neighbours_accepted = self.neighbours_acceptance(temperature)

        # Create the new state recipes and effects tensor
        # based on neighbour acception
        current_recipes = self.get_recipes()
        self.recipes = torch.where(
            neighbours_accepted.expand_as(current_recipes),
            self.neighbour_recipes,
            current_recipes
        )

        current_effects = self.get_effects()
        self.effects = torch.where(
            neighbours_accepted.expand_as(current_effects),
            self.neighbour_effects,
            current_effects
        )

    def mix_recipes(self, recipes: torch.Tensor):
        """Apply the recipes to compute effects tensor."""
        # Create the initial effects tensor
        effects = self.create_effects()

        # Mix each step of recipe
        for step in range(1, self.recipe_size + 1):
            # Fetch recipes and effects step
            recipe_step = recipes[step - 1, :].int()

            # Mix ingredients and store resultant effects tensor
            effects = self.mix_ingredients(recipe_step, effects)

        return effects
