from datetime import datetime
from recipe import Recipe


class Book:

    def __init__(self, name):
        assert name and isinstance(name, str),\
            "The value 'Name' is empty or is not a string."
        self.name = name
        self.last_update = datetime.now()
        self.creation_date = datetime.now()
        self.recipes_list = {'starter': [], 'lunch': [], 'dessert': []}

    def get_recipe_by_name(self, name):
        """Print a recipe with the name `name` and return the instance"""
        for key, item in self.recipes_list.items():
            for i in item:
                if i.name == name:
                    print(str(i))
                    return i
        print("The recipe with this name doesn't exist in this book")

    def get_recipes_by_types(self, recipe_type):
        """Get all recipe names for a given recipe_type """
        if recipe_type in self.recipes_list:
            newList = list()
            meals = self.recipes_list[recipe_type]
            for item in meals:
                newList.append(item.name)
                return newList
        else:
            print("This recipe type doesn't exist")

    def add_recipe(self, recipe):
        """Add a recipe to the book and update last_update"""
        assert recipe and isinstance(recipe, Recipe),\
            "The recipe is not a recipe"
        meal = recipe.recipe_type
        self.recipes_list[meal].append(recipe)
        self.last_update = datetime.now()
