import datetime
import string


def generator(text, sep=" ", option=None):
    """ - shuffle : mélange la liste des mots\n
        - unique : renvoie une liste dans laquelle chaque mot n'apparaît qu'une seule fois
        - ordered : trie les mots par ordre alphabétique"""
    try:
        assert type(text) is type('')
        assert type(sep) is type('')
    except:
        print("ERROR")
        exit()

    tab = []
    tab = text.split(sep)
    if (option == "ordered"):
        tab.sort()
        for i in tab:
            yield i
    elif (option == "unique"):
        j = []
        for i in tab:
            double = 0
            for k in j:
                if (k == i):
                    double += 1
            if (double == 0):
                j.append(i)
                yield i
    elif (option == "shuffle"):
        while len(tab):
            double = 0
            t = str(datetime.datetime.now())
            (h, seed) = t.split('.')
            rand = int(seed) % int(len(tab))
            yield tab[rand]
            tab.pop(rand)
    elif (option == None):
        for i in tab:
            yield i
    else:
        print("ERROR")
