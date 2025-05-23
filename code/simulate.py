"""Script to create Mixing Chain."""
import time
import torch
from tensors import Database, State
from typing import List, Tuple


class ChainSimulation(Database):
    """Class to simulate the mixing chain."""
    def _init__(self):
        """___init__."""
        # Instantiate Database
        super().__init__()

    def _encode_recipes(self, recipe: List[str]) -> torch.Tensor:
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

    def _decode_recipes(self, recipe: torch.Tensor):
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

    def optimize_recipes(
        self,
        base_product: str,
        n_batches: int,
        batch_size: int,
        n_steps: int,
        recipe_size: int = 7,
        initial_temperature: float = 1.0,
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            n_batches (int): Number of batches simulated.
            batch_size (int): Number of recipes in batch.
            distribution.
            n_steps (int): Number of steps simulation steps.
            recipe_size (int, optional): Number of ingredients in recipe.
            Defaults to 7.
            initial_temperature (float, optional): Initial value for Boltzmann temperature
            parameter. Defaults to 1.0.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Simulation parameters
        self.batch_size = batch_size
        self.recipe_size = recipe_size
        self.base_product = base_product
        self.initial_temperature = initial_temperature

        # Output tensors
        effects = torch.zeros(
            (n_steps, self.n_effects, n_batches * self.batch_size),
            dtype=torch.float32,
            device="cpu"
        )
        costs = torch.zeros(
            (n_steps, n_batches * self.batch_size),
            dtype=torch.float32,
            device="cpu"
        )
        values = torch.zeros(
            (n_steps, n_batches * self.batch_size),
            dtype=torch.float32,
            device="cpu"
        )

        with torch.no_grad():
            for b in range(n_batches):
                torch.cuda.empty_cache()
                # Define current state
                current_state = State(
                    base_product=base_product,
                    batch_size=self.batch_size,
                    recipe_size=self.recipe_size,
                )

                for t in range(n_steps):
                    temperature = (
                        initial_temperature /
                        torch.log(torch.tensor(t) + 1.01)
                    )
                    start_time = time.time()
                    print(f"Batch {b + 1}: Step {t + 1}")

                    # Evolve chain state
                    current_state.walk(temperature)

                    # Store effects
                    effects[
                        t, :, b * self.batch_size:(b + 1) * self.batch_size
                    ] = current_state.get_effects()

                    # Store costs
                    costs[
                        t, b * self.batch_size:(b + 1) * self.batch_size
                    ] = current_state.cost()

                    # Store values
                    values[
                        t, b * self.batch_size:(b + 1) * self.batch_size
                    ] = current_state.value()
                    print(f"TET: {round(time.time() - start_time, 3)}s")
                    t += 1

        # # Calculates objective
        # profits = values - costs

        # # Fetch optimal recipe
        # id_opt = torch.where(profits == profits.max())
        # opt_step, opt_sim = id_opt[0][0], id_opt[1][0]
        # recipe_opt = self._decode_recipes(recipes[:opt_step + 1, opt_sim])
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
