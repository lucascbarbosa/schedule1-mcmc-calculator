"""Script to create arrays used in simulation."""
import numpy as np
import pandas as pd
from typing import Tuple


class DatabaseArrays:
    """Class to create arrays from dataframes."""
    @staticmethod
    def convert_name_to_index(
        ingredients_df: pd.DataFrame,
        effects_df: pd.DataFrame,
        products_df: pd.DataFrame,
        rules_df: pd.DataFrame,
    ):
        """Convert ingredients, effects and products names to id.

        Args:
            ingredients_df (pd.DataFrame): Ingredients dataframe.
            effects_df (pd.DataFrame): Effects dataframe.
            products_df (pd.DataFrame): Products dataframe.
            rules_df (pd.DataFrame): Rules dataframe.
        """
        ingredient_to_index = ingredients_df.reset_index().set_index(
            "name")["index"]
        effect_to_index = effects_df.reset_index().set_index(
            "name")["index"]
        product_to_index = products_df.reset_index().set_index(
            "name")["index"]

        # Ingredients
        ingredients_df["ingredient_id"] = ingredients_df["name"].map(
            ingredient_to_index)
        ingredients_df["effect_id"] = ingredients_df["effect"].map(
            effect_to_index)
        ingredients_df = ingredients_df.rename(
            columns={"name": "ingredient_name", "effect": "effect_name"})

        # Effects
        effects_df["effect_id"] = effects_df["name"].map(effect_to_index)
        effects_df = effects_df.rename(columns={"name": "effect_name"})

        # Products
        products_df["product_id"] = products_df["name"].map(product_to_index)
        products_df["effect_id"] = products_df["effect"].map(effect_to_index)
        products_df = products_df.rename(
            columns={"name": "product_name", "effect": "effect_name"})

        # Rules
        rules_df["ingredient"] = rules_df[
            "ingredient"].map(ingredient_to_index)
        rules_df["effect_base"] = rules_df[
            "effect_base"].map(effect_to_index)
        rules_df["effect_result"] = rules_df[
            "effect_result"].map(effect_to_index)

        return ingredients_df, effects_df, products_df, rules_df

    @staticmethod
    def create_ingredients_arrays(
        ingredients_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create ingredients arrays."""
        ingredients_effect = ingredients_df['effect_id'].\
            to_numpy().reshape((1, -1))

        ingredients_cost = ingredients_df['cost'].to_numpy().reshape((1, -1))
        return (
            ingredients_effect,
            ingredients_cost,
        )

    @staticmethod
    def create_effects_arrays(
        effects_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create effects arrays."""
        effects_multiplier = effects_df['multiplier'].\
            to_numpy().reshape((1, -1))
        return effects_multiplier

    @staticmethod
    def create_effects_rules_matrix(
        ingredients_df: pd.DataFrame,
        effects_df: pd.DataFrame,
        rules_df: pd.DataFrame
    ) -> np.ndarray:
        """Create rules of mixing effects matrix.

        This is also a sparse binary matrix of shape (n_ingredients x n_effects
        x n_effects) indicating the effects conversion for each ingredient.

        """
        rules = np.array([
            np.eye(len(effects_df), dtype=int)
            for _ in range(len(ingredients_df))
        ])
        for _, row in rules_df.iterrows():
            rules[
                row["ingredient"],
                row["effect_result"],
                row["effect_base"]
            ] = 1
            rules[
                row["ingredient"],
                row["effect_base"],
                row["effect_base"]
            ] = 0
        return rules
