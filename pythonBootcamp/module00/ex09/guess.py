import random


def you_win(number, trials):
    if (number == 42):
        print("The answer to the ultimate question of life, the universe and everything is 42.")
    if (trials == 1):
        print("Congratulations! You got it on your first try!")
    else:
        print(f"Congratulation, you've got it!\nYou won in {trials} attempts!")


print(f"""\
This is an interactive guessing game!
You have to enter a number between 1 and 99 to find out the secret number.
Type 'exit' to end the game.
Good luck!
""")

trials = 1
number = random.randint(1, 99)
while True:
    answer = input("What's your guess between 1 and 99?\n")
    try:
        if answer == "exit":
            print("Goodbye!")
            break
        if int(answer) > number:
            trials += 1
            print("Too high!")
        elif int(answer) < number:
            trials += 1
            print("Too low!")
        else:
            you_win(number, trials)
            break

    except:
        print("That's not a number.")
        trials += 1
