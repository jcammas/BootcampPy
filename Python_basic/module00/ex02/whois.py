import sys

string = ''
i = len(sys.argv)
if (i > 1):
    if (i == 2):
        if (sys.argv[1].isdigit()):
            nbr = int(sys.argv[1])
            if (nbr == 0):
                print("I'm Zero.")
            elif (nbr % 2):
                print("I'm Odd.")
            else:
                print("I'm Even.")
        else:
            print("Error")
    else:
        print("Error")
