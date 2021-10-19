
import sys

ingredients = {	'sandwich': ['ham', 'bread', 'cheese', 'tomatoes'],
                'cake': ['flour', 'sugar', 'eggs'],
                'salad': ['avocado', 'arugula', 'tomatoes', 'spinach']}
cookbook = {'sandwich': {'ingredients': ingredients['sandwich'], 'type': 'lunch', 'time': 10},
            'cake': {'ingredients': ingredients['cake'], 'type': 'dessert', 'time': 60},
            'salad': {'ingredients': ingredients['salad'], 'type': 'lunch', 'time': 15}}


def valid_recipe_arg(recipe_name=''):
    while (len(recipe_name) <= 0
            or not recipe_name.isalpha()):
        recipe_name = input("Invalid parameter\n>> ")
    return recipe_name


def add_recipe():
    recipe_name = input(
        "\nname of the recipe :\n>> ")
    recipe_name = valid_recipe_arg(recipe_name)
    ingredients_list = []
    is_exit = ''
    while not (is_exit == 'exit'):
        ingredient = input("ingredient\n>> ")
        ingredients_list.append(valid_recipe_arg(ingredient))
        is_exit = input(
            "\nIngredient added, 'exit' if you want to leave.\n>> ")
        if (is_exit == 'exit'):
            break
    ingredients[recipe_name] = ingredients_list
    recipe_type = input("\ntype of meal ?\n>> ")
    recipe_type = valid_recipe_arg(recipe_type)

    recipe_timer = input("\nTime of cooking?\n>> ")
    while (not recipe_timer.isnumeric()
           or int(recipe_timer) < 0):
        recipe_timer = input("\nInvalid parameter\n>> ")
    cookbook[recipe_name] = {'ingredients': ingredients[recipe_name],
                             'type': recipe_type, 'time': int(recipe_timer)}


def get_recipe():
    recipe_name = ''
    while (recipe_name not in cookbook.keys()):
        print("\nPlease enter a valid recipe such as :", cookbook.keys())
        recipe_name = input(">> ")

    return recipe_name


def print_recipe(recipe_name):
    print("\nRecipe for " + str(recipe_name) + ':')
    print("\ningredients list: ", cookbook[recipe_name]['ingredients'])
    print("\nTo be eaten for ", cookbook[recipe_name]['type'])
    print("\ntakes ", cookbook[recipe_name]['time'], " of cooking.\n")


def delete_recipe(recipe_name):
    del cookbook[recipe_name]
    del ingredients[recipe_name]
    print(cookbook)


def main():
    while (1):
        print("Please select an option by typing the corresponding number: \n1: Add a recipe\n2: Delete a recipe\n"
              + "3: Print a recipe\n4: Print the cookbook\n5: Quit"
              )
        opt = input(">> ")
        while (not opt.isnumeric() or (int(opt) < 1 or int(opt) > 6)):
            opt = input(
                "\nThis option does not exist, please type the corresponding number.\nTo exit, enter 5.\n>> ")
        opt = int(opt)
        if (opt == 1):
            add_recipe()
        elif (opt == 2):
            recipe_name = get_recipe()
            delete_recipe(recipe_name)
        elif (opt == 3):
            recipe_name = get_recipe()
            print_recipe(recipe_name)
        elif (opt == 4):
            print(cookbook)
        elif (opt == 5):
            sys.exit("\nCookbook closed")


main()
