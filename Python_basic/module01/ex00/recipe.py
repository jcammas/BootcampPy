class Recipe:

    meal = ['starter', 'lunch', 'dessert']

    def __init__(
        self,
        name,
        cooking_lvl,
        cooking_time,
        ingredients,
        recipe_type,
        description=''
    ):
        assert name and isinstance(name, str),\
            "The value 'Name' is empty or is not a string."
        assert cooking_lvl and isinstance(cooking_lvl, int)\
            and cooking_lvl <= 5 and cooking_lvl >= 1,\
            "The value 'cooking_lvl' is not an int between 1 and 5."
        assert cooking_time and isinstance(cooking_time, int),\
            "The value 'cooking_time' is not a int."
        assert ingredients and isinstance(ingredients, list),\
            "The value 'ingredients' is not a list."
        for i in ingredients:
            assert i and isinstance(i, str),\
                "An item in the ingredient list is not a string."
        assert isinstance(description, str),\
            "The value 'description' is not a string."
        assert recipe_type and isinstance(recipe_type, str)\
            and recipe_type in self.meal,\
            "The value 'recipe_type' is not a string or"\
            "is not equal at 'starter', 'lunch' or 'dessert'."
        self.name = name
        self.cooking_lvl = cooking_lvl
        self.cooking_time = cooking_time
        self.ingredients = ingredients
        self.description = description
        self.recipe_type = recipe_type

    def __str__(self):
        """Return the string to print with the recipe info"""
        txt = "The name of the recipe: {name}"
        txt += "\nThe recipe is level {lvl},"
        txt += "\nRecipe takes about {time} minutes,"
        txt += "\nIngredients list: {ingredients},"
        txt += "\nTo be eaten for {meal}."
        txt += "\nThe description of the recipe: {description}"
        return txt.format(
            name=self.name,
            lvl=self.cooking_lvl,
            time=self.cooking_time,
            ingredients=self.ingredients,
            meal=self.recipe_type,
            description=self.description
        )
