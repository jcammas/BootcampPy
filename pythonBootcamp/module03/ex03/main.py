from ImageProcessor import ImageProcessor
from ScrapBooker import ScrapBooker
from ColorFilter import ColorFilter

imp = ImageProcessor()
arr = imp.load("../resources/elon.png")
scrap = ScrapBooker()
filter = ColorFilter()
# arr = scrap.juxtapose(arr, 10, 0)
# arr = scrap.thin(arr, 2, 1)
arr = filter.invert(arr)
imp.display(arr)
