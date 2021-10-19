class recipe:
    def __str__(self):
        txt = ""
        txt += "Name = "
        txt += self.name
        txt += "\n"
        txt += "Cooking level = "
        txt += str(self.cooking_lvl)
        txt += "\n"
        txt += "Ingredients = "
        #####
        txt += "Description = "
        txt += str(self.description)
        txt += "\n"
        txt += "Recipe type = "
        txt += str(self.recype_type)
        txt += "\n"
        return txt
