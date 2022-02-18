
from sys import argv as av


def print_usage():
    print(f"""Usage: python operations.py <number1> <number2>
Example:
  python operations.py 10 3""")


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def divide(a, b):
    if (b != 0):
        return a / b
    else:
        return "ERROR (div by zero)"


def modulo(a, b):
    if (b != 0):
        return a % b
    else:
        return "ERROR (modulo by zero)"


def print_results():
    try:
        sum = add(int(av[1]), int(av[2]))
        diff = sub(int(av[1]), int(av[2]))
        product = mul(int(av[1]), int(av[2]))
        quot = divide(int(av[1]), int(av[2]))
        rem = modulo(int(av[1]), int(av[2]))
    except:
        print("InputError: only numbers\n")
        print_usage()
        return

    print(f"""\
Sum:          {sum}
Difference:   {diff}
Product:      {product}
Quotient:     {quot}
Remainder:    {rem}""")


if (len(av) > 3):
    print("InputError: too many arguments\n")
    print_usage()
elif (len(av) == 3):
    print_results()
elif (len(av) < 3):
    print_usage()
