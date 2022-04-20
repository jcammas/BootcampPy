
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class ImageProcessor:

    def __init__(self):
        pass

    def load(self, path: str):
        """opens the .png file specified by the path argument and returns an
        array with the RGB values of the image pixels It must display a message specifying
        the dimensions of the image (e.g. 340 x 500)"""
        if not isinstance(path, str):
            raise ValueError(f"Invalid argument type -> str={type(path)}")
        img = Image.open(path)
        width, height = img.size
        pixels_tab = np.array(img)
        print("... Image : {} x {} ... Loading ...".format(width, height))
        return pixels_tab

    def display(self, array):
        """ takes a numpy array as an argument and displays the corresponding RGB image."""
        arr = array.reshape(array.shape)
        plt.imshow(arr)
        plt.show()


if __name__ == '__main__':
    imp = ImageProcessor()
    arr = imp.load("../resources/42AI.png")
    imp.display(arr)
    print(arr)
