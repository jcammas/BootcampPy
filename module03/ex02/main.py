import numpy as np
from ScrapBooker import ScrapBooker

spb = ScrapBooker()
arr1 = np.arange(0, 25).reshape(5, 5)

print("\n\nScrapBooker valid crop method : \n\n", spb.crop(arr1, (3, 1), (1, 0)))

arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1, 9)
print("\n\nScrapBooker valid thin method : \n\n", spb.thin(arr2, 3, 0))

arr3 = np.array([[var] * 10 for var in "ABCDEFG"])
print("\n\nScrapBooker valid thin method : \n\n", spb.thin(arr3, 3, 1))

arr4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n\nScrapBooker valid juxtapose method :\n\n", spb.juxtapose(arr4, 2, 0))

print("\n\n------- ERROR MANAGEMENT SHOULD PRINT NONE EVERYTIME--------")

arr4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

not_numpy_arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("\nScrapBooker unvalid crop method : \n",
      spb.crop(not_numpy_arr, (1, 2)))
print("\nScrapBooker unvalid juxtapose method :\n", spb.juxtapose(arr4, -2, 0))
print("\nScrapBooker unvalid mosaic method :\n", spb.mosaic(arr4, (1, 2, 3)))
