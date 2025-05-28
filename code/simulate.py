"""Script to create Mixing Chain."""
import torch
from tensors import Database, State
from tqdm import trange
from typing import List, Tuple


class ChainSimulation(Database):
    """Class to simulate the mixing chain."""
    def __init__(self):
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
        id_list = recipe.ravel().tolist()
        id_to_name = self.ingredients_df.set_index(
            "ingredient_id")["ingredient_name"].to_dict()
        names = []
        for ingredient_id in id_list:
            if ingredient_id < self.n_ingredients:
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
        n_simulations: int,
        n_steps: int,
        initial_temperature: float,
        alpha: float = 0.99,
        recipe_size: int = 7,
        objective_function: str = "profit"
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            n_simulations (int): Number of recipes simulated ate the same time.
            n_steps (int): Number of steps simulation steps.
            initial_temperature (float): Initial value for Boltzmann
            temperature. Defaults to 1.0.
            alpha (float, optional): Boltzmann temperature geometric factor.
            recipe_size (int, optional): Number of ingredients in recipe.
            Defaults to 7.
            objective_function (str, optional): Objective function used for
            Boltzmann distribution. Defaults to `profit`.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Output tensors
        sim_recipes = torch.zeros(
            (n_steps, recipe_size, n_simulations),
            dtype=torch.float32,
            device=self.device
        )
        sim_effects = torch.zeros(
            (n_steps, self.n_effects, n_simulations),
            dtype=torch.float32,
            device=self.device
        )
        sim_costs = torch.zeros(
            (n_steps, n_simulations),
            dtype=torch.float32,
            device=self.device
        )
        sim_values = torch.zeros(
            (n_steps, n_simulations),
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            torch.cuda.empty_cache()
            # Define current state
            current_state = State(
                base_product=base_product,
                n_simulations=n_simulations,
                recipe_size=recipe_size,
                objective_function=objective_function,
            )
            temperature = torch.tensor(
                initial_temperature,
                dtype=torch.float32,
                device=self.device
            )
            for t in trange(n_steps, desc="Steps", leave=False):
                # Store recipes
                sim_recipes[t, :, :] = current_state.get_recipes()

                # Store effects
                sim_effects[t :, :] = current_state.get_effects()

                # Store costs
                sim_costs[t, :] = current_state.cost()

                # Store values
                sim_values[t, :] = current_state.value()

                # Evolve chain state
                current_state.walk(temperature)

                # Decreases temperature with geometric schedule
                temperature *= alpha

        # Calculates profit
        profits = sim_values - sim_costs

        # Fetch optimal recipe
        id_opt = torch.where(profits == profits.max())
        opt_step, opt_sim = id_opt[0][0], id_opt[1][0]

        # Correct recipe and cost
        recipe_opt = current_state.correct_recipe(
            sim_recipes[opt_step, :, opt_sim]
        )
        cost_opt = float(current_state.cost(recipe_opt))

        # Extract original value
        value_opt = float(sim_values[opt_step, opt_sim])

        # Calculates new profit
        profit_opt = value_opt - cost_opt

        # Decode recipe and effects tensors
        recipe_opt = self._decode_recipes(recipe_opt)
        effects_opt = self._decode_effects(sim_effects[opt_step, :, opt_sim])

        results_data = {
            "recipes": sim_recipes,
            "effects": sim_effects,
            "costs": sim_costs,
            "values": sim_values,
            "profits": profits
        }
        results_opt = {
            "recipe": recipe_opt,
            "effects": effects_opt,
            "cost": cost_opt,
            "value": value_opt,
            "profit": profit_opt
        }

        return results_data, results_opt

    def mix_recipe(self, base_product: str, recipe: List[str]) -> dict:
        """Mix a list of ingredients."""
        # Define state
        state = State(
            base_product=base_product,
            n_simulations=1,
            recipe_size=len(recipe),
            objective_function='profit',
            create_recipes=False
        )

        # Encode recipe
        recipe = self._encode_recipes(recipe).to(device=self.device)
        state.recipes = recipe

        # Mix recipe
        state.effects = state.mix_recipes(recipe)

        # Correct recipe and cost
        recipe = state.correct_recipe(recipe)
        cost = float(state.cost(recipe))

        # Fetch original value
        value = float(state.value())

        # Calculates profit
        profit = value - cost

        # Decode recipe and effects tensors
        recipe = self._decode_recipes(recipe)
        effects = self._decode_effects(state.effects)

        return {
            "recipe": recipe,
            "effects": effects,
            "cost": cost,
            "value": value,
            "profit": profit,
        }
