from book import Book
from recipe import Recipe

if __name__ == "__main__":
    print("test 1: create valide recipe", end="\n\n")
    r = Recipe(
        "Gato banane",
        2,
        25,
        ['200g flour', '30g sugar', '3 eggs',
            '20g butter', '25cl milk', 'des bananes'],
        'dessert',
        "gato banane"
    )
    print(r, end="\n\n")
    print("test 2: create book", end="\n\n")
    b = Book("Le book a banane")
    print("book create at:", b.creation_date)
    print("book update at:", b.last_update, end="\n\n")
    print("test 3: add valide recipe in book", end="\n\n")
    print("book recipes list before add:", b.recipes_list)
    b.add_recipe(r)
    print("book recipes list after add:", b.recipes_list)
    print("book update at:", b.last_update, end="\n\n")
    print("test 4: print all recipes by type of dessert", end="\n\n")
    print(b.get_recipes_by_types('dessert'), end="\n\n")
    print("test 5: print all recipes by type of lunch", end="\n\n")
    print(b.get_recipes_by_types('lunch'), end="\n\n")
    print("test 6: get recipe by name", end="\n\n")
    newR = b.get_recipe_by_name('Gato banane')
    print("newR id:", id(newR), end="\n\n")
