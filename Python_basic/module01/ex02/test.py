from vector import Vector


if __name__ == "__main__":
    print("Test 1 create Vector with list: [0.0, 1.0, 2.0, 3.0]")
    v1 = Vector([0.0, 1.0, 2.0, 3.0])
    print(v1, end="\n\n")
    print("Test 2 create Vector with int: 3")
    v2 = Vector(3)
    print(v2, end="\n\n")
    print("Test 3 create Vector with tuple: (10, 15)")
    v3 = Vector((10, 15))
    print(v3, end="\n\n")
    print("Test 4: v1 (test1) + 5")
    v4 = v1 + 5
    print(v4, end="\n\n")
    print("Test 5: 5 + v1")
    v5 = 5 + v1
    print(v5, end="\n\n")
    print("Test 6: Vector((5, 9)) + v1")
    v6 = Vector((5, 9)) + v1
    print(v6, end="\n\n")
    print("Test 7: v1 + Vector((5, 9))")
    v7 = v1 + Vector((5, 9))
    print(v7, end="\n\n")
    print("Test 8: v1 - 42")
    v8 = v1 - 42
    print(v8, end="\n\n")
    print("Test 9: v2 - Vector([5.0, 10.0, 42.0])")
    v9 = v2 - Vector([5.0, 10.0, 42.0])
    print(v9, end="\n\n")
    print("Test 10: Vector([5.0, 10.0, 42.0]) - v2")
    v10 = Vector([5.0, 10.0, 42.0]) - v2
    print(v10, end="\n\n")
    print("Test 11: v2 / v9")
    v11 = v2 / v9
    print(v11, end="\n\n")
    print("Test 12: v9 / v2")
    v12 = v9 / v2
    print(v12, end="\n\n")
    print("Test 13: v3 * Vector((40, 45))")
    v13 = v3 * Vector((40, 45))
    print(v13, end="\n\n")
    print("Test 14: Vector((40, 45)) * v3")
    v14 = Vector((40, 45)) * v3
    print(v14, end="\n\n")
