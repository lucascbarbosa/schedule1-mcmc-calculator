"""Script to create Mixing Chain."""
from tensors import DatabaseTensors, StateTensors
import time
import torch
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
            else:
                raise ValueError(f"Ingredient name {n} is incorrect!")
        return torch.tensor(recipe_encoded).reshape((len(recipe), 1))

    def _encode_effects(self, effects: List[str]) -> torch.Tensor:
        """Convert from effects name to id."""
        name_to_id = self.effects_df.set_index(
            "effect_name")["effect_id"].to_dict()
        effects_encoded = []
        for n in effects:
            if n in name_to_id:
                effects_encoded.append(name_to_id[n])
            else:
                raise ValueError(f"Effect name {n} is incorrect!")
        return torch.tensor(effects_encoded).reshape((len(effects), 1))

    def _decode_recipe(self, recipe: torch.Tensor):
        """Convert from ingredients id to name."""
        id_list = recipe.tolist()
        id_to_name = self.ingredients_df.set_index(
            "ingredient_id")["ingredient_name"].to_dict()
        return [id_to_name[i] for i in id_list]

    def _decode_effects(self, effects: torch.Tensor):
        """Convert from effects id to name."""
        effects_id = torch.where(effects == 1)[0]
        return self.effects_df[
            self.effects_df["effect_id"].isin(effects_id.tolist())
        ]["effect_name"].tolist()

    def _boltzmann_temperature(self, step: int):
        """Calculate temperature parameter of Boltzmann distribution."""
        return self.T0 / torch.log(torch.tensor(step + 1.01))

    def _neighbours_acceptance(
        self,
        current_state: StateTensors,
        neighbours_state: StateTensors,
        step: int,
    ) -> torch.Tensor:
        """Calculates probability of choosing each ingredient.

        This is a Boltzmann probability adjusted by the Metropoles-Hastings
        acceptance parameter that takes into account the profit resulting from
        adding that ingredient. The temperature parameter is computed via
        simulated annealing with log schedule.
        """
        # Compute profits
        current_profit = current_state.profit()
        neighbours_profit = neighbours_state.profit().reshape((
            self.n_ingredients, self.batch_size
        ))

        # Fetch current temperature parameter via log schedule
        T = self._boltzmann_temperature(step)
        acceps = torch.clamp(
            torch.exp((neighbours_profit - current_profit) / T), max=1.0
        )
        return acceps

    def compute_ingredient_prob(
        self,
        state: StateTensors,
        neighbours_state: StateTensors,
        step: int
    ) -> torch.Tensor:
        """Compute adjusted probability of using each ingredient.

        Given a current state, the neighbour states are the results
        from adding an ingredient to the product.
        """
        # Generate all possible ingredients tensor.
        all_ingredients = torch.arange(
            self.n_ingredients
        ).unsqueeze(1).expand(-1, self.batch_size).reshape((
            self.batch_size * self.n_ingredients
        ))

        # Copy current state tensors
        ingredients_count, active_effects = state.get_tensors()

        # Set neighbours state tensors
        neighbours_ingredients_count = ingredients_count.repeat(
            1, self.n_ingredients)
        neighbours_active_effects = active_effects.repeat(
            1, self.n_ingredients)
        neighbours_state.set_tensors(
            neighbours_ingredients_count,
            neighbours_active_effects
        )

        # Mix ingredient
        neighbours_state.mix_ingredient(
            ingredients=all_ingredients
        )

        # Get neighbours acceptance
        neighbours_acceptance = self._neighbours_acceptance(
            current_state=state,
            neighbours_state=neighbours_state,
            step=step
        )

        # Calculate probability from normal adjusted by acceptance
        neighbours_prob = (
            neighbours_acceptance /
            neighbours_acceptance.sum(dim=0)
        )
        return neighbours_prob

    def optimize_recipe(
        self,
        base_product: str,
        num_simulations: int,
        batch_size: int,
        num_steps: int = 8,
        T0: float = 1.0,
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            num_simulations (int): Number of batches simulated.
            batch_size (int): Number of recipes in batch.
            num_steps (int, optional): Number of simulation steps.
            Defaults to 8.
            T0 (float, optional): Initial value for Boltzmann temperature
            parameter. Defaults to 1.0.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Simulation parameters
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.base_product = base_product
        self.T0 = T0

        # Output tensors
        recipes = torch.zeros(
            num_steps, num_simulations * batch_size, dtype=torch.long,
            device=self.device)
        effects = torch.zeros(
            num_steps, num_simulations * batch_size, self.n_effects,
            dtype=torch.long, device=self.device
        )
        costs = torch.zeros(
            num_steps, num_simulations * batch_size, dtype=torch.long,
            device=self.device)
        values = torch.zeros(
            num_steps, num_simulations * batch_size, dtype=torch.long,
            device=self.device)

        # Define neighbours state
        neighbours_state = StateTensors(
            self.base_product,
            self.batch_size,
            torch_device=self.device
        )

        with torch.no_grad():
            for s in range(num_simulations):
                state = StateTensors(
                    base_product=base_product,
                    batch_size=batch_size,
                    torch_device=self.device
                )
                for t in range(num_steps):
                    start_time = time.time()
                    print(f"Batch simulation {s + 1}: Step {t + 1}")
                    # Ingredients choice probability (n_ingredients x batch_size)
                    ingredients_probs = self.compute_ingredient_prob(
                        state,
                        neighbours_state,
                        t
                    )
                    ingredients = torch.multinomial(
                        ingredients_probs.T, num_samples=1).squeeze(1)

                    # Mix ingredients to state
                    state.mix_ingredient(ingredients=ingredients)

                    # Store ingredient in recipe
                    recipes[
                        t, s * batch_size:(s + 1) * batch_size] = ingredients
                    effects[
                        t, s * batch_size:(s + 1) * batch_size, :
                    ] = state.active_effects.T
                    costs[
                        t, s * batch_size:(s + 1) * batch_size] = state.cost()
                    values[
                        t, s * batch_size:(s + 1) * batch_size] = state.value()
                    print(f"TET: {round(time.time() - start_time, 2)}")

        # Calculates profits
        profits = values - costs

        # Fetch optimal recipe
        id_opt = torch.where(profits == profits.max())
        opt_step, opt_sim = id_opt[0][0], id_opt[1][0]
        recipe_opt = self._decode_recipe(recipes[:opt_step + 1, opt_sim])
        effects_opt = self._decode_effects(effects[opt_step, opt_sim, :])
        cost_opt = float(costs[opt_step, opt_sim])
        value_opt = float(values[opt_step, opt_sim])
        profit_opt = float(profits[opt_step, opt_sim])
        results_data = {
            "recipes": recipes,
            "effects": effects,
            "costs": costs,
            "values": values,
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

    def mix_recipe(
        self,
        base_product: str,
        recipe: List[str]
    ) -> Tuple[List[str], float, float, float]:
        """Create a mix given a recipe.

        Args:
            base_product (str): Base product.
            recipe (List): List with order of ingredients to be mixed.

        Returns:
            Tuple[list, float, float, float]:
            Optimal recipe with effects, cost, value and profit.

        Raises:
            ValueError: if ingredient name in recipe is incorrect.
        """
        self.batch_size = 1
        # Set base state
        state = StateTensors(
            base_product=base_product,
            batch_size=1,
            torch_device=self.torch_device
        )

        # Encode ingredients in recipe
        recipe = self._encode_recipe(recipe)
        for i in range(recipe.shape[0]):
            ingredient = recipe[i]
            # Mix ingredients to state
            state.mix_ingredient(ingredients=ingredient)

        effects = self._decode_effects(state.active_effects[:, 0])
        cost = float(state.cost())
        value = float(state.value())
        profit = value - cost
        return {
            "effects": effects,
            "cost": cost,
            "value": value,
            "profit": profit
        }
