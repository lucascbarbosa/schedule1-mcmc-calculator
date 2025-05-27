"""Script to mix a custom recipe."""
from simulate import ChainSimulation


chain = ChainSimulation()
base_product = 'Cocaine'
recipe = [
    'Cuke',
    'Cuke',
]
results = chain.mix_recipe(base_product, recipe)
print(f"Base Product: {base_product}")
print(f"Recipe: {results['recipe']}")
print(f"Effects: {results['effects']}")
print(f"Cost: {results['cost']}")
print(f"Value: {results['value']}")
print(f"Profit: {results['profit']}")
