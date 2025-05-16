"""Script to create Mixing Chain."""
from tensors import DatabaseTensors, StateTensors
import torch
from tqdm import trange
from typing import List, Tuple


class ChainSimulation(DatabaseTensors):
    """Class to simulate the mixing chain."""
    def _init__(self):
        """___init__."""
        # Instantiate DatabaseTensors
        super().__init__()

    def decode_effects(self, effects: torch.Tensor):
        """Convert effects from id to name."""
        effects_id = torch.where(effects == 1)[0]
        return self.effects_df[
            self.effects_df["effect_id"].isin(effects_id.tolist())
        ]["effect_name"].tolist()

    def decode_recipe(self, recipe: torch.Tensor):
        """Convert ingredients from id to name."""
        id_list = recipe.tolist()
        id_to_name = self.ingredients_df.set_index(
            "ingredient_id")["ingredient_name"].to_dict()
        return [id_to_name[i] for i in id_list]

    def _neighbour_acceptance(
        self,
        current_state: StateTensors,
        neighbour_state: StateTensors,
        step: int,
    ) -> torch.Tensor:
        """Calculates probability of choosing each ingredient.

        This is a Boltzmann probability adjusted by the Metropoles-Hastings
        acceptance parameter that takes into account the profit resulting from
        adding that ingredient. The temperature parameter is computed via
        simulated annealing with log schedule.
        """
        # Acceptance parameter
        neighbour_profit = (
            neighbour_state.value() - neighbour_state.cost()
        ).ravel()
        current_profit = current_state.value() - current_state.cost()

        # Fetch current temperature parameter via log schedule
        T = self.T0 / torch.log(torch.tensor(step + 1.0))
        acceps = torch.clamp(
            torch.exp((neighbour_profit - current_profit) / T), max=1.0
        )
        return acceps

    def compute_ingredient_prob(
        self, state: StateTensors, step: int) -> torch.Tensor:
        """Compute adjusted probability of using each ingredient.

        Given a current state, the neighbour states are the results
        from adding an ingredient to the product.
        """
        # The number of neighbours is equal to the number of ingredients,
        # since the ingredient is the edge that induces the state transition.
        n_neighbours = self.n_ingredients

        # Generate all possible ingredients tensor.
        all_ingredients = torch.arange(
            self.n_ingredients
        ).unsqueeze(1).expand(-1, self.batch_size)

        # Store neighbours state tensors
        neighbours_acceptances = torch.zeros((
            n_neighbours,
            self.batch_size
        ))

        for i in range(n_neighbours):
            ingredients = all_ingredients[i, :]

            # Copy state
            ingredients_count, active_effects = state.get_tensors()
            neighbour_state = StateTensors(
                self.base_product, self.batch_size
            )
            neighbour_state.set_tensors(ingredients_count, active_effects)

            # Mix ingredient
            neighbour_state.mix_ingredient(
                ingredients=ingredients
            )

            # Calculate probabilities
            neighbours_acceptances[i, :] = self._neighbour_acceptance(
                current_state=state,
                neighbour_state=neighbour_state,
                step=step,
            )

        neighbours_probs = (
            neighbours_acceptances / neighbours_acceptances.sum(dim=0)
        )
        return neighbours_probs

    def optimize_recipe(
        self,
        base_product: str,
        batch_size: int,
        num_steps: int = 8,
        T0: float = 1.0,
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            batch_size (int): Number of simulations in batch.
            num_steps (int, optional): Max number of simulation steps.
            Defaults to 8.
            T (float, optional): Boltzmann temperature. Defaults to 1.0.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Simulation parameters
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.base_product = base_product
        self.T0 = T0

        state = StateTensors(base_product=base_product, batch_size=batch_size)

        # Output tensors
        recipes = torch.zeros(
            batch_size, num_steps, dtype=torch.long, device=self.device)

        for t in trange(num_steps, desc="Batch simulation"):
            # Ingredients choice probability (n_ingredients x batch_size)
            ingredients_probs = self.compute_ingredient_prob(state, t)
            ingredients = torch.multinomial(
                ingredients_probs.T, num_samples=1).squeeze(1)

            # Mix ingredients to state
            state.mix_ingredient(ingredients=ingredients)

            # Store ingredient in recipe
            recipes[:, t] = ingredients

        # Calculates profits
        effects = state.active_effects.T
        costs = state.cost()
        values = state.value()
        profits = values - costs

        # Fetch optimal recipe
        id_opt = torch.where(profits == profits.max())[1][0]
        recipe_opt = self.decode_recipe(recipes[id_opt, :])
        effects_opt = self.decode_effects(effects[id_opt, :])
        cost_opt = float(costs[0, id_opt])
        value_opt = float(values[0, id_opt])
        profit_opt = float(profits[0, id_opt])
        return {
            "recipe": recipe_opt,
            "effects": effects_opt,
            "cost": cost_opt,
            "value": value_opt,
            "profit": profit_opt
        }

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
        def encode_recipe(recipe: List[str]) -> torch.Tensor:
            """Convert ingredient names to ids."""
            name_to_id = self.ingredients_df.set_index(
                "ingredient_name")["ingredient_id"].to_dict()
            recipe_encoded = []
            for n in recipe:
                if n in name_to_id:
                    recipe_encoded.append(name_to_id[n])
                else:
                    raise ValueError(f"Ingredient name {n} is incorrect!")
            return torch.tensor(recipe_encoded).reshape((len(recipe), 1))

        self.batch_size = 1
        # Set base state
        state = StateTensors(base_product=base_product, batch_size=1)

        # Encode ingredients in recipe
        recipe = encode_recipe(recipe)
        for i in range(recipe.shape[0]):
            ingredient = recipe[i]
            # Mix ingredients to state
            state.mix_ingredient(ingredients=ingredient)

        effects = self.decode_effects(state.active_effects[:, 0])
        cost = float(state.cost())
        value = float(state.value())
        profit = value - cost
        return {
            "effects": effects,
            "cost": cost,
            "value": value,
            "profit": profit
        }


chain = ChainSimulation()
# recipe = ['Horse S*men', 'Motor Oil', 'Paracetamol']
# recipe = [
#     'Cuke',
#     'Energy Drink',
#     'Horse S*men',
#     'Banana',
#     'Horse S*men',
# ]
# recipe = [
#     'Horse S*men',
#     'V*agra',
#     'Mega Bean',
#     'Donut',
#     'Iodine',
#     'Donut',
#     'Battery'
# ]
# recipe = ['Iodine', 'Flu Medicine', 'Horse S*men', 'Mouth Wash', 'Banana', 'Horse S*men', 'Iodine', 'Banana']
# results = chain.mix_recipe("OG Kush", recipe)
# print(f"Receita: {recipe}\nEfeitos: {results['effects']}.\nCusto: {results['cost']}\nValor: {results['value']}")

results = chain.optimize_recipe(
    "OG Kush", batch_size=100_000, num_steps=8, T0=5.0)
print(
f"""
OTIMIZADO:
Receita: {results["recipe"]}
Efeitos: {results["effects"]}
Custo: {results["cost"]}
Valor: {results["value"]}
Profit: {results["profit"]}
"""
)
