import re

cookbook = {
    "sandwich": {
        "ingredients": ["ham", "bread", "cheese", "tomatoes"],
        "meal": "lunch",
        "prep_time": 10,
    },
    "cake": {
        "ingredients": ["flour", "sugar", "eggs"],
        "meal": "dessert",
        "prep_time": 60,
    },
    "salad": {
        "ingredients": ["avocado", "arugula", "tomatoes", "spinach"],
        "meal": "lunch",
        "prep_time": 15,
    },
}


def first_question():
    for key in cookbook.keys():
        print(key)
    for value in cookbook.values():
        print(value)
    for item in cookbook.items():
        print(item)


def print_recipe(name_of_the_recipe):
    try:
        print(f"""\
Recipe for {name_of_the_recipe}:
Ingredients list: {cookbook[name_of_the_recipe]["ingredients"]}
To be eaten for {cookbook[name_of_the_recipe]["meal"]}.
Takes {cookbook[name_of_the_recipe]["prep_time"]} minutes of cooking.""")
    except:
        print("Recipe does not exist !")


def delete_recipe(name_of_the_recipe):
    try:
        del cookbook[name_of_the_recipe]
    except:
        print("Recipe does not exist !")


def create_recipe(name_of_recipe, ingredients, meal, prep_time):
    try:
        cookbook[str(name_of_recipe)] = {
            "ingredients": [ingredient for ingredient in ingredients],
            "meal": str(meal),
            "prep_time": int(prep_time),
        }
    except:
        print("Oops, somethng went wrong !")


def print_all_recipes():
    key_list = ["%i: %s" % (index + 1, value)
                for index, value in enumerate(cookbook.keys())]
    print("\n".join(key_list))


def quit_cookbook():
    print("Cookbook closed.")


def input_for_create():
    try:
        name_of_recipe = str(input("Name of the new recipe:"))
        ingredients = re.findall(
            r"[\w']+", str(input("Name of all ingredients seperated with a coma (ie: Salt, honey, frog):")))
        meal = str(input("Type of meal:"))
        prep_time = int(input("Preparation time (in minutes):"))
    except:
        print("Wrong input !")
    create_recipe(name_of_recipe, ingredients, meal, prep_time)


def input_for_delete():
    try:
        delete_recipe(str(input("Which recipe should we delete?:")))
    except:
        print("Wrong input !")


def input_for_print():
    try:
        print_recipe(
            str(input("Please enter the recipe's name to get its details:")))
    except:
        print("Wrong input !")


def menu():
    menu_items = [
        ("Add a recipe", input_for_create),
        ("Delete a recipe", input_for_delete),
        ("Print a recipe", input_for_print),
        ("Print the cookbook", print_all_recipes),
        ("Quit", quit_cookbook),
    ]
    entries = ["%i: %s" % (index + 1, value[0])
               for index, value in enumerate(menu_items)]
    print("Please select an option by typing the corresponding number:")
    print("\n".join(entries))
    while (True):
        try:
            choice = int(input())
            menu_items[choice - 1][1]()
            if (choice == 5):
                break
        except:
            print(f"""\
This option does not exist, please type the corresponding number.
To exit, enter 5.""")


# first_question()
menu()
