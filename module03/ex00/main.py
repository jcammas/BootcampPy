from NumPyCreator import NumPyCreator

npc = NumPyCreator()
ret = npc.from_list([[1, 2, 3], [6, 3, 4]])
print("from_list : " + str(ret))
ret = npc.from_tuple(("a", "b", "c"))
print("from_tuple : " + str(ret))
# ret = npc.from_iterable(range(5))
print("from_iterable : " + str(ret))
shape = (3, 5)
ret = npc.from_shape(shape)
print("from_shape : " + str(ret))
ret = npc.random(shape)
print("random : " + str(ret))
ret = npc.identity(9)
print("identity : " + str(ret))


print(npc.from_list("toto"))
print(npc.from_list([[1, 2, 3], [6, 3, 4], [8, 5, 6, 7]]))
print(npc.from_tuple(3.2))
print(npc.from_tuple(((1, 5, 8), (7, 5))))
print(npc.from_shape((-1, -1)))
print(npc.identity(-1))
