"""Script to create Mixing Chain."""
import torch
from tensors import DatabaseTensors
from tqdm import trange
from typing import List, Tuple


class ChainState(DatabaseTensors):
    """Arrays related to chain state for batched simulation.

    The ingredients array maps the ingredients used in the recipe, i.e.,
    the path traveled in the chain.

    The effects array indicates the active effects, i.e., the state of
    the chain.

    """
    def __init__(
        self,
        base_product: str,
        batch_size: int,
        active_effects: torch.Tensor = [],
        ingredients_count: torch.Tensor = [],
    ):
        """Create ingredients and effects tensors.

        Args:
            base_product (str): Base product used as source state.
            batch_size (int): Number of simulations in batch.

        Raises:
            ValueError: If chosen based product.
        """
        # Instantiate DatabaseTensors
        super().__init__()

        # Get base product cost, value and effect
        base_product = self.products_df[
            self.products_df["product_name"] == base_product]
        if len(base_product) == 0:
            raise ValueError(f"Base product {base_product} is invalid!")
        self.product_value = base_product["value"].iloc[0]

        if len(active_effects) == 0 and len(ingredients_count) == 0:
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

        else:
            self.active_effects = active_effects
            self.ingredients_count = ingredients_count

    def cost(self) -> float:
        """Calculates the cost of current state."""
        return (
            self.ingredients_cost @ self.ingredients_count
        )

    def value(self) -> float:
        """Calculates sell value of current state."""
        mult = self.effects_multiplier @ self.active_effects
        return (1 + mult) * float(self.product_value)

    def increment_ingredient_count(
        self,
        ingredients: torch.Tensor,
        ingredients_count: torch.Tensor = None,
    ) -> torch.Tensor:
        """Add mixed ingredient to the count."""
        if ingredients_count is None:
            ingredients_count = self.ingredients_count.clone()
        ingredients_count[
            ingredients,
            torch.arange(ingredients.shape[0])
        ] += 1.0
        return ingredients_count

    def apply_ingredients_effect(
        self,
        ingredients: torch.Tensor,
        active_effects: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply effect from added ingredients."""
        if active_effects is None:
            active_effects = self.active_effects.clone()
        if active_effects.sum(dim=0).all() < 8:
            active_effects[
                self.ingredients_effect[0, ingredients].to(int),
                torch.arange(ingredients.shape[0])
            ] = 1
        return active_effects

    def apply_effects_rules(
        self,
        ingredients: torch.Tensor,
        active_effects: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply effects transition rules for each ingredient."""
        if active_effects is None:
            active_effects = self.active_effects.clone()

        # Apply rules for the current ingredients
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

        return effects_result

    def mix_ingredient(self, ingredients: torch.Tensor):
        """Mix products with ingredients."""
        #  Increment ingredient count
        self.ingredients_count = self.increment_ingredient_count(
            ingredients=ingredients
        )

        # print(ingredients)
        # print(self.active_effects)
        # Apply effects transition rules
        self.active_effects = self.apply_effects_rules(
            ingredients=ingredients
        )

        # print(self.active_effects)
        # Apply ingredient effect
        self.active_effects = self.apply_ingredients_effect(
            ingredients=ingredients
        )
        # print(self.active_effects)
        # print("\n\n")


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
        current_state: ChainState,
        neighbour__active_effects: torch.Tensor,
        neighbour__ingredients_count: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates probability of choosing each ingredient.

        This is a uniform probability adjusted by the Metropoles-Hastings
        acceptance parameter that takes into account the profit resulting from
        adding that ingredient.
        """
        # Acceptance parameter
        neighbour_state = ChainState(
            base_product=self.base_product,
            batch_size=self.batch_size,
            ingredients_count=neighbour__ingredients_count,
            active_effects=neighbour__active_effects
        )
        neighbour_profit = (
            neighbour_state.value() - neighbour_state.cost()
        ).ravel()
        current_profit = current_state.value() - current_state.cost()
        acceps = torch.exp(
            torch.clamp(neighbour_profit / current_profit, max=0.0)
        )
        return acceps

    def compute_ingredient_prob(self, state: ChainState) -> torch.Tensor:
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
            # Generate a possible ingredients tensor
            ingredients = all_ingredients[i, :]

            #  Increment ingredient count
            neighbour__ingredients_count = state.\
                increment_ingredient_count(
                    ingredients=ingredients
                )

            # Apply effects transition rules to neighbours
            neighbour__active_effects = state.apply_effects_rules(
                ingredients=ingredients
            )

            # Apply ingredient effect
            neighbour__active_effects = state.\
                apply_ingredients_effect(
                    ingredients=ingredients,
                    active_effects=neighbour__active_effects,
                )

            # Calculate probabilities
            neighbours_acceptances[i, :] = self._neighbour_acceptance(
                current_state=state,
                neighbour__active_effects=neighbour__active_effects,
                neighbour__ingredients_count=neighbour__ingredients_count)

        neighbours_probs = (
            neighbours_acceptances / neighbours_acceptances.sum(dim=0)
        )
        return neighbours_probs

    def optimize_recipe(
        self,
        base_product: str,
        batch_size: int,
        num_steps: int = 8,
    ) -> Tuple[List[str], List[str], float, float, float]:
        """Run parallelized simulation.

        Args:
            base_product (str): Base product.
            batch_size (int): Number of simulations in batch.
            num_steps (int, optional): Max number of simulation steps.
            Defaults to 8.

        Returns:
            Tuple[List[str], List[str], float, float, float]:
            Optimal recipe with effects, cost, value and profit.
        """
        # Simulation parameters
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.base_product = base_product

        state = ChainState(base_product=base_product, batch_size=batch_size)

        # Output tensors
        recipes = torch.zeros(
            batch_size, num_steps, dtype=torch.long, device=self.device)

        for t in trange(num_steps, desc="Batch simulation"):
            # Ingredients choice probability (n_ingredients x batch_size)
            ingredients_probs = self.compute_ingredient_prob(state)
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
        id_opt = torch.where(profits == profits.max())[0][0]
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
        state = ChainState(base_product=base_product, batch_size=1)

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
recipe = [
    'Cuke',
    'Energy Drink',
    'Horse S*men',
    'Banana',
    'Horse S*men',
]
# recipe = [
#     'Horse S*men',
#     'V*agra',
#     'Mega Bean',
#     'Donut',
#     'Iodine',
#     'Donut',
#     'Battery'
# ]
results = chain.mix_recipe("OG Kush", recipe)
print(f"Receita: {recipe}\nEfeitos: {results['effects']}.\nCusto: {results['cost']}\nValor: {results['value']}")

# results = chain.optimize_recipe("OG Kush", batch_size=5, num_steps=5)

# print(
# f"""
# OTIMIZADO:
# Receita: {results["recipe"]}
# Efeitos: {results["effects"]}
# Custo: {results["cost"]}
# Valor: {results["value"]}
# Profit: {results["profit"]}
# """
# )
