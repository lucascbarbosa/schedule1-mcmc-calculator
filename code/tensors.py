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
        num_steps: int,
    ):
        """Create initial ingredients and effects tensors.

        Args:
            base_product (str): Base product used as source state.
            batch_size (int): Number of simulations in batch.
            torch_device (torch.device): Torch device (cpu or cuda).
            num_steps (int): Number of steps in simulation.

        Raises:
            ValueError: If chosen based product.
        """
        # Define global variables
        super().__init__(torch_device=torch_device)
        self.batch_size = batch_size
        self.num_steps = num_steps

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
            (self.n_ingredients, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )

        # Create a sparse binary effects array.
        # Earch element of the array indicates if effect is present.
        self.active_effects = torch.zeros(
            (self.n_effects, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )
        self.active_effects[
            self.effects_df["effect_name"] ==
            base_product["effect_name"].iloc[0], :
        ] = 1

        # Path of ingredients count and active effects
        self.ingredients_count_path = torch.zeros(
            (self.num_steps + 1, self.n_ingredients, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )
        self.ingredients_count_path[0, :, :] = self.ingredients_count.clone()
        self.active_effects_path = torch.zeros(
            (self.num_steps + 1, self.n_effects, self.batch_size),
            dtype=torch.float32,
            device=self.device
        )
        self.active_effects_path[0, :, :] = self.active_effects.clone()

        # Path length
        self.path_length = torch.zeros(
            self.batch_size,
            dtype=torch.int32,
            device=self.device
        )

        # Previous ingredients count and active effects
        self.previous_ingredients_count = self.ingredients_count.clone()
        self.previous_active_effects = self.active_effects.clone()

    def get_current_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current ingredients and effects tensors."""
        return self.ingredients_count.clone(), self.active_effects.clone()

    def get_previous_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get previous ingredients and effects tensors."""
        return (
            self.previous_ingredients_count.clone(),
            self.previous_active_effects.clone()
        )

    def current_cost(self) -> float:
        """Calculates the cost of current state."""
        return (
            self.ingredients_cost @ self.ingredients_count
        )

    def previous_cost(self) -> float:
        """Calculates the cost of previous state."""
        return (
            self.ingredients_cost @ self.previous_ingredients_count
        )

    def current_value(self) -> float:
        """Calculates sell value of current state."""
        mult = self.effects_multiplier @ self.active_effects
        return (1 + mult) * float(self.product_value)

    def previous_value(self) -> float:
        """Calculates sell value of previous state."""
        mult = self.effects_multiplier @ self.previous_active_effects
        return (1 + mult) * float(self.product_value)

    def current_profit(self) -> float:
        """Calculates profit of current state."""
        return self.current_value() - self.current_cost()

    def previous_profit(self) -> float:
        """Calculates profit of previous state."""
        return self.previous_value() - self.previous_cost()

    def current_effects_distance(self, desired_effects: torch.Tensor) -> float:
        """Distance current state active and desired effects."""
        return (
            -torch.sum((self.active_effects - desired_effects)**2, dim=0)
        )

    def previous_effects_distance(self, desired_effects: torch.Tensor) -> float:
        """Distance previous state active and desired effects."""
        return (
            -torch.sum(
                (self.previous_active_effects - desired_effects)**2,
                dim=0
            )
        )

    def increment_ingredient_count(
        self,
        ingredients: torch.Tensor,
    ) -> torch.Tensor:
        """Add mixed ingredient to the count."""
        # Clone tensors
        ingredients_count = self.ingredients_count.clone()

        # Increment ingredients count
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
        # Clone tensors
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
        # Clone current and past effects
        active_effects = self.active_effects.clone()
        # shape: (n_effects, batch_size)

        # Fetch rules from each added ingredient
        rules_batch = self.rules[ingredients]
        # shape: (batch_size, n_effects, n_effects)

        # Compute rules
        effects_result = torch.bmm(
            rules_batch,
            active_effects.T.unsqueeze(2)
        ).squeeze(2).T  # shape: (n_effects, batch_size)

        # Detect all duplicates effects (> 1.0) and clamp them back to 1.0
        duplicated_mask = effects_result > 1.0
        effects_result = effects_result.clamp_max(1.0)

        # Restore wrongly deactivated effects
        if duplicated_mask.any():
            # Get duplicated effects
            effect_ids, ingredient_ids = duplicated_mask.nonzero(as_tuple=True)
            active_sources = (
                rules_batch.bool() & active_effects.T.unsqueeze(2).bool()
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

        self.active_effects = effects_result

    def mix_ingredient(self, ingredients: torch.Tensor):
        """Mix products with ingredients."""
        ingredients = ingredients.clone()

        # If ingredient has to be removed, set it to 0 (hardcoded)
        remove_ingredients_mask = (
            ingredients == self.n_ingredients).to(self.device)
        ingredients[remove_ingredients_mask] = 0
        # Ids of ingredients to be removed and added
        removed_ingredients_ids = torch.nonzero(
            remove_ingredients_mask,
            as_tuple=False).ravel()
        added_ingredients_ids = torch.nonzero(
            ~remove_ingredients_mask,
            as_tuple=False).ravel()

        # Fetch pre mix state tensors
        (
            pre_mix_ingredients_count, pre_mix_active_effects
        ) = self.get_current_tensors()

        #  Increment ingredient count
        self.increment_ingredient_count(ingredients)

        # Apply effects transition rules
        self.apply_effects_rules(ingredients)

        # Apply ingredient effect
        self.apply_ingredients_effect(ingredients)

        # Increment path length
        self.path_length[added_ingredients_ids] += 1
        self.path_length[removed_ingredients_ids] -= 1
        self.path_length = torch.clamp(
            self.path_length, min=0, max=self.num_steps
        )

        # Restore previous state for any ingredients equal to 0
        if removed_ingredients_ids.shape[0] > 0:
            # Get previous state tensors
            (
                previous_ingredients_count, previous_active_effects
            ) = self.get_previous_tensors()

            # Restore previous ingredients count for removed ingredients
            self.ingredients_count_path[
                self.path_length[removed_ingredients_ids] - 1,
                :,
                removed_ingredients_ids
            ] = 0.0
            self.ingredients_count[
                :,
                removed_ingredients_ids
            ] = previous_ingredients_count[:, removed_ingredients_ids]

            # Restore previous active effects for removed ingredients
            self.active_effects_path[
                self.path_length[removed_ingredients_ids] - 1,
                :,
                removed_ingredients_ids
            ] = 0.0
            self.active_effects[
                :,
                removed_ingredients_ids
            ] = previous_active_effects[:, removed_ingredients_ids]

        if added_ingredients_ids.shape[0] > 0:
            # Updated paths
            self.ingredients_count_path[
                self.path_length[added_ingredients_ids],
                :,
                added_ingredients_ids
            ] = pre_mix_ingredients_count[
                :, added_ingredients_ids
            ].T

            self.active_effects_path[
                self.path_length[added_ingredients_ids],
                :,
                added_ingredients_ids
            ] = pre_mix_active_effects[
                :, added_ingredients_ids
            ].T

        # Update previous state
        self.previous_ingredients_count = self.ingredients_count_path[
            self.path_length, :, torch.arange(self.path_length.shape[0])].T

        self.previous_active_effects = self.active_effects_path[
            self.path_length, :, torch.arange(self.path_length.shape[0])].T
