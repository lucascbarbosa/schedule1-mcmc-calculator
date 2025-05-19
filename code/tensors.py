"""Script to create tensors used in simulation."""
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple


class DatabaseTensors:
    """Class to create tensors from dataframes."""
    def __init__(self, torch_device: str):
        """Convert ingredients, effects and products names to id."""
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
        self.device = torch_device

        # Fetch ingredients arrays
        self.create_ingredients_tensors()

        # Fetch effects multiplier tensors
        self.create_effects_tensors()

        # Fetch rules tensor
        self.create_effects_rules_tensor()

    def create_ingredients_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create ingredients tensors."""
        self.ingredients_effect = torch.from_numpy(
            self.ingredients_df['effect_id'].to_numpy().reshape((1, -1))
        ).to(dtype=torch.float32, device=self.device)

        self.ingredients_cost = torch.from_numpy(
            self.ingredients_df['cost'].to_numpy().reshape((1, -1))
        ).to(dtype=torch.float32, device=self.device)

    def create_effects_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
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


class StateTensors(DatabaseTensors):
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
        torch_device: str,
    ):
        """Create initial ingredients and effects tensors.

        Args:
            base_product (str): Base product used as source state.
            batch_size (int): Number of simulations in batch.
            torch_device (torch.device): Torch device (cpu or cuda).

        Raises:
            ValueError: If chosen based product.
        """
        # Define global variables
        super().__init__(torch_device=torch_device)
        self.batch_size = batch_size

        # Get base product cost, value and effect
        base_product = self.products_df[
            self.products_df["product_name"] == base_product]
        if len(base_product) == 0:
            raise ValueError(f"Base product {base_product} is invalid!")
        self.product_value = base_product["value"].iloc[0]

        # Create an integer ingredients array.
        # Each element of the array represents how many times that
        # ingredient was used.
        self.ingredients_count = torch.zeros(
            (self.n_ingredients, batch_size),
            dtype=torch.float32,
            device=self.device
        )

        # Create a sparse binary effects array.
        # Earch element of the array indicates if effect is present.
        self.active_effects = torch.zeros(
            (self.n_effects, batch_size),
            dtype=torch.float32,
            device=self.device
        )
        self.active_effects[
            self.effects_df["effect_name"] ==
            base_product["effect_name"].iloc[0], :
        ] = 1

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ingredients and effects tensors."""
        return self.ingredients_count.clone(), self.active_effects.clone()

    def set_tensors(
        self,
        ingredients_count: torch.Tensor,
        active_effects: torch.Tensor
    ):
        """Set ingredients and effects tensors."""
        self.ingredients_count = ingredients_count
        self.active_effects = active_effects

    def cost(self) -> float:
        """Calculates the cost of current state."""
        return (
            self.ingredients_cost @ self.ingredients_count
        )

    def value(self) -> float:
        """Calculates sell value of current state."""
        mult = self.effects_multiplier @ self.active_effects
        return (1 + mult) * float(self.product_value)

    def profit(self) -> float:
        """Calculates profit of current state."""
        return self.value() - self.cost()

    def effects_distance(self, desired_effects: torch.Tensor) -> float:
        """Distance from active and desired effects."""
        return torch.sum((self.active_effects - desired_effects)**2)

    def increment_ingredient_count(
        self,
        ingredients: torch.Tensor,
    ) -> torch.Tensor:
        """Add mixed ingredient to the count."""
        ingredients_count = self.ingredients_count.clone()
        ingredients_count[
            ingredients,
            torch.arange(ingredients.shape[0])
        ] += 1.0
        self.ingredients_count = ingredients_count

    def apply_ingredients_effect(
        self,
        ingredients: torch.Tensor,
    ) -> torch.Tensor:
        """Apply effect from added ingredients."""
        active_effects = self.active_effects.clone()
        # Check if sum of effects is leq 8
        sum_effects = active_effects.sum(dim=0)
        sum_mask = sum_effects < 8
        # Apply ingredient effects only if sum of effects is leq 8
        effect_ids = self.ingredients_effect[0, ingredients]
        filtered_batch_ids = torch.nonzero(sum_mask).squeeze(1)
        filtered_effect_ids = effect_ids[sum_mask]
        active_effects[filtered_effect_ids.to(int), filtered_batch_ids] = 1
        self.active_effects = active_effects

    def apply_effects_rules(
        self,
        ingredients: torch.Tensor,
    ) -> torch.Tensor:
        """Apply effects transition rules for each ingredient."""
        # Apply rules for the current ingredients
        active_effects = self.active_effects.clone()
        rules_batch = self.rules[ingredients]
        effects_result = torch.bmm(
            rules_batch,
            active_effects.T.unsqueeze(2)
        ).squeeze(2).T

        # Clamp duplicated effects to 1.0
        duplicated = effects_result == 2.0
        effects_result[duplicated] = 1.0

        # Restore wrongly deactivated effects
        for i in range(ingredients.shape[0]):
            duplicated_ids = torch.nonzero(
                duplicated[:, i], as_tuple=False).flatten()
            for target_id in duplicated_ids:
                source_ids = torch.nonzero(
                    rules_batch[i, target_id, :] == 1.0,
                    as_tuple=False
                ).flatten()
                for source_id in source_ids:
                    if (
                        active_effects[source_id, i] == 1.0 and
                        effects_result[source_id, i] == 0.0
                    ):
                        effects_result[source_id, i] = 1.0

        self.active_effects = effects_result

    def mix_ingredient(self, ingredients: torch.Tensor):
        """Mix products with ingredients."""
        #  Increment ingredient count
        self.increment_ingredient_count(
            ingredients=ingredients
        )

        # Apply effects transition rules
        self.apply_effects_rules(
            ingredients=ingredients
        )

        # Apply ingredient effect
        self.apply_ingredients_effect(
            ingredients=ingredients
        )
