from cmath import log
import time
import logging
from random import randint
import os

logging.basicConfig(filename='machine.log', filemode='w', level=logging.INFO)


def log(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        logging.info("(cmaxime)Running: {}{} [ exec-time = {} ms ]".format(
            fn.__name__, (20 - len(fn.__name__)) * ' ',  str(end - start)[0:5]))
        return result
    return wrapper


class CoffeeMachine():
    water_level = 100

    @log
    def start_machine(self):
        if self.water_level > 20:
            return True
        else:
            print("Please add water!")
            return False

    @log
    def boil_water(self):
        return "boiling..."

    @log
    def make_coffee(self):
        if self.start_machine():
            for _ in range(20):
                time.sleep(0.1)
                self.water_level -= 1
            print(self.boil_water())
            print("Coffee is ready!")

    @log
    def add_water(self, water_level):
        time.sleep(randint(1, 5))
        self.water_level += water_level
        print("Blub blub blub...")


if __name__ == "__main__":
    machine = CoffeeMachine()
    for i in range(0, 5):
        machine.make_coffee()
    machine.make_coffee()
    machine.add_water(70)
# https://docs.python.org/3/howto/logging.html
