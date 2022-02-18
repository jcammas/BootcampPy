from book import Book
from recipe import Recipe

# instantiation de l'objet recette "quiche"
quiche = Recipe("LA QUICHE", 3, 45, [
                "sel", "poivre", "crème", "des trucs"], "Faire un appareil de quiche et roule !", "lunch")
choux = Recipe("Le choux", 3, 45, ["choux", "poivre", "crème"],
               "Faire bouillir le choux, ajouter les symboles normands!", "lunch")
print("Impression de la recette pour s'assurer de sa création\n")
print(str(choux))

# Création du livre test_book
test_book = Book("ZeLivre")

# Ajout de la recette de quice au livre
test_book.add_recipe(quiche)
test_book.add_recipe(choux)

print("\n-------------------------------\nTest get_recipes_by_types avec 'lunch'\n")
test_book.get_recipes_by_types("lunch")

print("\n-------------------------------\nTest get_recipes_by_name avec 'LA QUICHE'\n")
test_book.get_recipe_by_name("LA QUICHE")

print(
    f"date de création du livre {test_book.creation_date}\ndernière update {test_book.last_update}")
