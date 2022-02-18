from numpy import isin


class Account(object):

    ID_COUNT = 1

    def __init__(self, name, **kwargs):
        self.id = self.ID_COUNT
        self.name = name
        self.__dict__.update(kwargs)
        Account.ID_COUNT += 1

    def transfer(self, amount):
        self.value += amount

    def ohnanpaslui(self):
        valid = True
        attributes = list(self.__dict__.keys())
        if "value" not in attributes:
            return True
        if len(attributes) % 2 == 0:
            return True
        for i in attributes:
            if i[0] == "b":
                return True
            if i.startswith("zip") or i.startswith("addr"):
                valid = False
        return valid


class Bank(object):
    """The bank"""

    def __init__(self):
        self.account = []

    def add(self, account):
        self.account.append(account)

    def getlecompte(self, c):
        if isinstance(c, int):
            for i in self.account:
                if i.id == c:
                    return i
        if isinstance(c, str):
            for i in self.account:
                if i.name == c:
                    return i

    def transfer(self, origin, dest, amount: float) -> bool:
        expediteur = self.getlecompte(origin)
        destinataire = self.getlecompte(dest)
        if not isinstance(expediteur, Account) or not isinstance(destinataire, Account):
            return False
        elif not expediteur.ohnanpaslui() and not destinataire.ohnanpaslui():
            value = expediteur.value
            if value >= amount and value > 0:
                expediteur.value -= amount
                destinataire.transfer(amount)
                return True
        return False

    def fix_account(self, c) -> bool:
        account = self.getlecompte(c)
        if isinstance(account, Account):
            if account.ohnanpaslui():
                attributes = list(account.__dict__.keys())
                if 'addr' not in attributes:
                    account.__dict__.update({'zip': ''})
                if 'addr' not in attributes:
                    account.__dict__.update({'addr': ''})
                if 'name' not in attributes:
                    account.__dict__.update({'name': c})
                if 'value' not in attributes:
                    account.__dict__.update({'value': 0})

                for i in attributes:
                    if i.startswith('b'):
                        tmp = i
                        while i.startswith('b'):
                            tmp = tmp[1:]
                        account.__dict__[tmp] = account.__dict__[i]
                        del account.__dict__[i]

                if account.__dict__.__len__() % 2 == 0:
                    return False
        if account.ohnanpaslui():
            return False
        return True
