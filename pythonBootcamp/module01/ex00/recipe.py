class Recipe:

    def __init__(self, name, cooking_lvl, cooking_time, ingredients, description="", recipe_type=None):
        if type(name) is not str:
            raise ValueError("invalid name!")
            return None
        if type(cooking_lvl) is not int or cooking_lvl < 1 or cooking_lvl > 5:
            raise ValueError("invalid cooking level!")
            return None
        if type(cooking_time) is not int or cooking_lvl < 0:
            raise ValueError("invalid cooking time!")
            return None
        if type(ingredients) is not list:
            raise ValueError("invalid ingredients list!")
            return None
        for ingredient in ingredients:
            if type(ingredient) is not str:
                raise ValueError("invalid ingredient!")
                return None
        if type(recipe_type) is not str or recipe_type not in ["starter", "lunch", "dessert"]:
            raise ValueError("invalid recipe type!")
            return None
        if description != "" and type(description) is not str:
            raise ValueError("invalid description!")
            return None

        self.name = name
        self.cooking_lvl = cooking_lvl
        self.cooking_time = cooking_time
        self.ingredients = ingredients
        self.description = description
        self.recipe_type = recipe_type

    def __str__(self):
        """Return the string to print with the recipe info"""
        txt = f"""\
        {self.name}
Type de plat: {self.recipe_type}
Niveau de difficulté: {self.cooking_lvl}
Temps de cuisson: {self.cooking_time}
Ingrédients: {self.ingredients}
{self.description}"""

        return txt
