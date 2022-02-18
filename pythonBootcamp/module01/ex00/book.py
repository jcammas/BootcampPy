from recipe import Recipe
from datetime import date, datetime


class Book:
    def __init__(self, name):
        assert name and isinstance(
            name, str), "The value 'Name' is empty or is not a string."
        self.name = name
        self.creation_date = datetime.now()
        self.last_update = datetime.now()
        self.recipes_list = {
            'starter': [],
            'lunch': [],
            'dessert': [],
        }

    def get_recipe_by_name(self, name):
        """Print a recipe with the name `name` and return the instance"""
        for recipes in self.recipes_list.values():
            for recipe in recipes:
                if recipe.name == name:
                    print(str(recipe))
                    return recipe
        print("Not found. Try another recipe name")

    def get_recipes_by_types(self, recipe_type):
        """Get all recipe names for a given recipe_type"""
        if (recipe_type not in self.recipes_list.keys()):
            raise ValueError("invalid recipe type!")
            return None
        for recipe in self.recipes_list[recipe_type]:
            print(recipe.name)

    def add_recipe(self, recipe):
        """Add a recipe to the book and update last_update"""
        assert recipe and isinstance(
            recipe, Recipe), "The value 'recipe' is empty or is not a Recipe object."
        self.recipes_list[recipe.recipe_type].append(recipe)
        self.last_update = datetime.now()
