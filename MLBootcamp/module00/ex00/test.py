from matrix import Matrix, Vector

m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
             [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0],
             [6.0, 7.0]])
m3 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0],
             [6.0, 7.0]])

m4 = Matrix([[5, 5, 5, 5], [5, 5, 5, 5]])
m5 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m6 = Matrix([[0.0, 1.0, 2.0],
             [0.0, 2.0, 4.0]])

v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])

print("---- ADD -----")
print("")
print("")
print(m1 + m4)
# print(m1 + m3)
print("")
print("")
print("---- SUB -----")
print("")
print("")
print(m1 - m4)
# print(m1 - m3)
print("")
print("")
print("---- DIV -----")
print("")
print("")
print(m3 / 2)
print(m3 / 0)
print("")
print("")

print("---- MULT -----")
print("")
print("")
print(m3 * 2)
print(m1 * m2)
print(m6 * v1)
print("")
print("")
print("---- VECTOR ADD -----")
print("")
print("")
print(v1 + v2)

print("")
print("")

print("----- SHAPE / T -----")
print("")
print("")
print(m5.shape)
print(m5.T())
print(m5.T().shape)
print("")
print("")
