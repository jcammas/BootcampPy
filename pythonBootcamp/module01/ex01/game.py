class GotCharacter:

    def __init__(self, first_name=None, is_alive=True):
        assert first_name and isinstance(first_name, str), "wrong first_name"
        assert is_alive and isinstance(is_alive, bool), "wrong is_alive"
        self.first_name = first_name
        self.is_alive = is_alive


class Lannister(GotCharacter):
    """Class representing the Lannister family."""

    def __init__(self, first_name=None, is_alive=True):
        super().__init__(first_name=first_name, is_alive=is_alive)
        self.family_name = "Lannister"
        self.house_words = "Hear Me Roar!"

    def die(self):
        if self.is_alive == True:
            self.is_alive = False

    def print_house_words(self):
        print(self.house_words)
