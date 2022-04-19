import sys


def check_error(ingredients):
    valid = map(lambda x: True if (type(x) is str and len(x) > 2)
                else False, ingredients)
    if False in valid:
        return False
    return True


class Recipe:
    def __init__(self, name=None, cooking_lvl=None, cooking_time=None,
                 ingredients=None, description=None, recipe_type=None):
        self.errors = []
        self.name = None
        self.cooking_lvl = None
        self.cooking_time = None
        self.ingredients = None
        self.recipe_type = None
        self.ft_name(name)
        self.ft_cooking_mama(cooking_lvl)
        self.ft_cooking_time(cooking_time)
        self.ft_igr(ingredients)
        self.description = description if type(description) is str else ''
        self.ft_recipe(recipe_type)
        self.is_ok()

    def is_ok(self):
        if len(self.errors) != 0:
            for i, error in enumerate(self.errors):
                sys.exit("Error")

    def ft_name(self, name):
        if not name:
            self.errors.append('Error')
            return
        elif type(name) is not str:
            self.errors.append('Error')
            return
        self.name = name

    def ft_cooking_mama(self, cooking_lvl):
        if not cooking_lvl:
            self.errors.append('Error')
            return
        elif type(cooking_lvl) is not int:
            self.errors.append('Error')
            return
        elif type(cooking_lvl) is int and (cooking_lvl < 1 or cooking_lvl > 5):
            self.errors.append('Error.')
        self.cooking_lvl = cooking_lvl

    def ft_cooking_time(self, cooking_time):
        if type(cooking_time) in [int, float] and cooking_time == 0:
            self.errors.append("Error")
            return
        elif not cooking_time:
            self.errors.append('Error')
            return
        elif type(cooking_time) not in [int, float]:
            self.errors.append('Error')
            return
        elif type(cooking_time) in [int, float] and (cooking_time < 0):
            self.errors.append("Error")
            return
        self.cooking_time = cooking_time

    def ft_igr(self, ingredients):
        if not ingredients and type(ingredients) is list:
            self.errors.append('Error')
            return
        elif type(ingredients) is not list:
            self.errors.append(
                f'Error')
            return
        elif not check_error(ingredients):
            self.errors.append('Error')
            return
        self.ingredients = ingredients

    def ft_recipe(self, recipe_type):
        if not recipe_type:
            self.errors.append('Error')
            return
        elif type(recipe_type) is not str:
            self.errors.append('Error')
            return
        elif recipe_type not in ['starter', 'lunch', 'dessert']:
            self.errors.append('Error')
            return
        self.recipe_type = recipe_type

    def __str__(self):
        return f"name : {self.name}\ncooking_lvl : {self.cooking_lvl}\ncooking_time: {self.cooking_time}\ningredients : {self.ingredients}\ndescription : {self.description}\nrecipe_type : {self.recipe_type}"
