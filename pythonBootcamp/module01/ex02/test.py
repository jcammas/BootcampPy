from vector import Vector


try:
    init = 5
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = -1
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = [1.0, 3.0, 5.0, 8.0]
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = [1.0, 3, 5, 8.0]
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = []
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = ["a", 2]
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = (3, 8)
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = (3, "a")
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = ("bbb", 8)
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = (4, 8, 10)
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    init = (8, 4)
    print(f"init with {init}:")
    v = Vector(init)
    print(f"{v}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")


try:
    a = Vector(7)
    b = 5
    print(f"do {a} + {b}:")
    print(f"{a + b}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = Vector([4.0, 2.0, 3.0, 6.0, 8.0, 9.0, 6.3])
    print(f"do {a} + {b}:")
    print(f"{a + b}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = Vector([4.0, 2.0, 3.0, 6.0, 8.0, 9.0])
    print(f"do {a} + {b}:")
    print(f"{a + b}\n************************\n")
except TypeError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = "Awowuouwuw"
    print(f"do {a} + {b}:")
    print(f"{a + b}\n************************\n")
except TypeError as msg:
    print(f"{msg}\n************************\n")


try:
    a = Vector(7)
    b = 5
    print(f"do {b} + {a}:")
    print(f"{b + a}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = Vector([4.0, 2.0, 3.0, 6.0, 8.0, 9.0, 6.3])
    print(f"do {b} + {a}:")
    print(f"{b + a}\n************************\n")
except ValueError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = Vector([4.0, 2.0, 3.0, 6.0, 8.0, 9.0])
    print(f"do {b} + {a}:")
    print(f"{b + a}\n************************\n")
except TypeError as msg:
    print(f"{msg}\n************************\n")

try:
    a = Vector(7)
    b = "Awowuouwuw"
    print(f"do {b} + {a}:")
    print(f"{b + a}\n************************\n")
except TypeError as msg:
    print(f"{msg}\n************************\n")
