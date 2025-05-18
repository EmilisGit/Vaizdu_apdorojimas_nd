import numpy as np
import cv2
import matplotlib.pyplot as plt

class Vaizdas:
    def __init__(self, path: str):
        self.path = path
        self.rgb = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
        self._original = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
        self.mask = None
        assert self._original is not None, f"\n\nImage: {path}, could not be loaded. \n\n"

    @property
    def original(self):
        return self._original  
    
    @property
    def gray(self):
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
    
    def show(self, image = None, cmap=None):
        if image is None:
            image = self.rgb
            
        fig, ax = plt.subplots()
        im = ax.imshow(image, cmap=cmap)
        ax.set_axis_off()
        print(f"Image shape: {image.shape}")
        plt.show()
    
    def histogram(self, channels=None):
        if channels is None:
            plt.hist(self.gray.ravel(),256,[0,256])
        else:
            colors = tuple(channels.lower())
            for i,col in enumerate(colors):
                histr = cv2.calcHist([self.rgb],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])

        plt.show()

    def set_mask(self, mask):
        self.mask = mask