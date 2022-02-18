import sys

if len(sys.argv) > 2:
    print("ERROR")
    sys.exit(1)
elif len(sys.argv) == 1:
    sys.exit(0)
try:
    nb = int(sys.argv[1])
except ValueError:
    print("ERROR")
    sys.exit(1)

if nb == 0:
    print("I'm Zero.")
elif nb % 2 is 0:
    print("I'm Even.")
else:
    print("I'm Odd.")
