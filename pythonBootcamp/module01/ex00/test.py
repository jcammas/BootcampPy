from book import Book
from recipe import Recipe

# print(Recipe("cooki", 0, 10, ["dough", "sugar",
#       "love"], "deliciousness incarnate", "dessert"))

# print(Recipe("cooki", 1.5, 10, ["dough", "sugar",
#       "love"], "deliciousness incarnate", "dessert"))

# print(Recipe("cooki", 1, 10, [], "deliciousness incarnate", "dessert"))

# Recipe("cooki", 1, 10, ["dough", "sugar", "love"],
#        "deliciousness incarnate", "dessert")
# print("Congratulations you finally made some delicous cookies")

b = Book("My seductive recipes")

print("\nshoud be the current date and time:")
print(b.creation_date)

print("\nshould be the same as the creation date or None")
print(b.last_update)

crumble = Recipe("Crumble", 1, 25, [
                 "apples", "flour", "sugar"], "delicious", "dessert")

b.add_recipe(crumble)
print("\nShould be a different date / time than the one printed before")
print(b.last_update)

print()
print(b.get_recipe_by_name("Crumble"))

print()
print(b.get_recipe_by_name("Liver Icecream"))

print()
print(b.get_recipes_by_types("dessert")[0])

print(b.get_recipes_by_types("asdasd"))
