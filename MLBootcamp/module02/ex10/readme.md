#### l'ex 10 du module 02

# step by step pour space_avocado.py

- Appliquer un data spliter sur space_avocado.csv pour avoir un training et un test set => éviter l'overfitting

- utiliser polynomial_features sur le set de training :

        def add_polynomial_features(x, power):
        """Add polynomial features to vector x by raising its values up to the power given in argument.
        Args:
        x: has to be an numpy.array, a vector of shape m _ 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
        Return:
        The matrix of polynomial features as a numpy.array, of shape m _ n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
        return None
        if not isinstance(power, int):
        return None

        # yˆ = θ0 + θ1\*x + θ2x**2 + · · · + θnx**n

        res = x
        for i in range(2, power + 1):
        tmp = x \*\* (i)
        res = np.concatenate((res, tmp), axis=1)
        return res

        && dans le main on l'utilise sur toutes nos pred


        for i in range(1, 5):
        print(f"{i}")
        X_poly = add_polynomial_features(X_, i)
        lr = model_load(i)

- réaliser 4 régression linéaire avec un entrainement à chaque fois :

        for i in range(1, 5):
        print(f"{i}")
        X_poly = add_polynomial_features(X_, i)
        lr = model_load(i)
        y_hat = lr.predict(X_poly)
        cost = lr.cost_(Y, y_hat)
        print(f"{cost = }")
        loss.append(cost)
        prd.append(y_hat)

- évaluer nos models

- plot nos models => quel est le meilleur modèle obtenu ? => le modèle 4

- plot la diff enntre vrai prix et meilleur prix obtenu par notre modèle

- enregistrer nos data dans des .pkl afin de les réutiliser dans benchmark_train :

            def model_load(poly):
            """In models.[csv/yml/pickle] one must find the parameters of all the
            models you have explored and trained. In space_avocado.py train the model based on
            the best hypothesis you find and load the other models from models.[csv/yml/pickle].
            Then evaluate and plot the different graphics as asked before.'"""
            path = os.path.join(os.path.dirname(__file__), f"model_{poly}.pkl")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
