"""Script to create Mixing Chain."""
import time
import torch
from tensors import DatabaseTensors, StateTensors
from typing import List, Tuple


class ChainSimulation(DatabaseTensors):
    """Class to simulate the mixing chain."""
    def _init__(self, torch_device: str):
        """___init__."""
        # Instantiate DatabaseTensors
        super().__init__(torch_device=torch_device)

    def _encode_recipe(self, recipe: List[str]) -> torch.Tensor:
        """Convert from ingredients name to id."""
        name_to_id = self.ingredients_df.set_index(
            "ingredient_name")["ingredient_id"].to_dict()
        recipe_encoded = []
        for n in recipe:
            if n in name_to_id:
                recipe_encoded.append(name_to_id[n])
            elif n == 'Remove':
                recipe_encoded.append(self.n_ingredients)
            else:
                raise ValueError(f"Ingredient name {n} is incorrect!")
        return torch.tensor(recipe_encoded).reshape((len(recipe), 1))

    def _encode_effects(self, effects: List[str]) -> torch.Tensor:
        """Convert from effects name to id."""
        name_to_id = self.effects_df.set_index(
            "effect_name")["effect_id"].to_dict()
        effects_encoded = torch.zeros(
            (self.n_effects, 1), device=self.device)
        for n in effects:
            if n in name_to_id:
                effects_encoded[name_to_id[n]] = 1.0
            else:
                raise ValueError(f"Effect name {n} is incorrect!")
        return effects_encoded

    def _decode_recipe(self, recipe: torch.Tensor):
        """Convert from ingredients id to name."""
        id_list = recipe.tolist()
        id_to_name = self.ingredients_df.set_index(
            "ingredient_id")["ingredient_name"].to_dict()
        names = []
        for ingredient_id in id_list:
            if ingredient_id != self.n_ingredients: 
                names.append(id_to_name[ingredient_id])
        return names

    def _decode_effects(self, effects: torch.Tensor):
        """Convert from effects id to name."""
        effects_id = torch.where(effects == 1)[0]
        return self.effects_df[
            self.effects_df["effect_id"].isin(effects_id.tolist())
        ]["effect_name"].tolist()

    def optimize_recipe(
        self,
        base_product: str,
        n_batches: int,
        batch_size: int,
        recipe_size: int = 7,
        T0: float = 1.0,
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            n_batches (int): Number of batches simulated.
            batch_size (int): Number of recipes in batch.
            distribution.
            recipe_size (int, optional): Number of ingredients in recipe.
            Defaults to 7.
            T0 (float, optional): Initial value for Boltzmann temperature
            parameter. Defaults to 1.0.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Simulation parameters
        self.batch_size = batch_size
        self.recipe_size = recipe_size
        self.base_product = base_product
        self.T0 = T0

        # Output tensors
        effects = torch.zeros(
            recipe_size, n_batches * self.batch_size, self.n_effects,
            dtype=torch.float32,
            device="cpu"
        )
        costs = torch.zeros(
            recipe_size, n_batches * self.batch_size,
            dtype=torch.float32,
            device="cpu"
        )
        values = torch.zeros(
            recipe_size, n_batches * self.batch_size,
            dtype=torch.float32,
            device="cpu"
        )

        with torch.no_grad():
            for b in range(n_batches):
                torch.cuda.empty_cache()
                # Define current state
                current_state = StateTensors(
                    base_product=base_product,
                    batch_size=self.batch_size,
                    torch_device=self.device,
                    recipe_size=self.recipe_size,
                )

                for t in range(recipe_size):
                    start_time = time.time()
                    print(f"Batch {b + 1}: Step {t + 1}")

                    # Create neighbours
                    neighbours_state = current_state.create_neighbours()

        #             # Store effects
        #             effects[
        #                 t, b * self.batch_size:(b + 1) * self.batch_size, :
        #             ] = current_state.active_effects.T

        #             # Store costs
        #             costs[
        #                 t, b * self.batch_size:(b + 1) * self.batch_size
        #             ] = current_state.current_cost()

        #             # Store values
        #             values[
        #                 t, b * self.batch_size:(b + 1) * self.batch_size
        #             ] = current_state.current_value()
        #             print(f"TET: {round(time.time() - start_time, 3)}s")
        #             t += 1

        # # Calculates objective
        # profits = values - costs

        # # Fetch optimal recipe
        # id_opt = torch.where(profits == profits.max())
        # opt_step, opt_sim = id_opt[0][0], id_opt[1][0]
        # recipe_opt = self._decode_recipe(recipes[:opt_step + 1, opt_sim])
        # effects_opt = self._decode_effects(effects[opt_step, opt_sim, :])
        # cost_opt = float(costs[opt_step, opt_sim])
        # value_opt = float(values[opt_step, opt_sim])
        # profit_opt = float(profits[opt_step, opt_sim])
        # results_data = {
        #     "recipes": recipes,
        #     "effects": effects,
        #     "costs": costs,
        #     "values": values,
        #     "profits": profits
        # }
        # results_opt = {
        #     "recipe": recipe_opt,
        #     "effects": effects_opt,
        #     "cost": cost_opt,
        #     "value": value_opt,
        #     "profit": profit_opt
        # }

        # return results_data, results_opt
