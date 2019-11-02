import os
from skimage.feature import daisy
from PIL import Image

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

import numpy as np


class DAISY(object):
    """the DAISY module"""

    def __str__(self):
        return "\nUsing the algorithm Daisy.....\n"

    def get_name(self):
        return "DAISY"   

    def get_options(self):
        cf = ConfigParser.ConfigParser()
        cf.read("config.cof")

        option_dict = dict()

        for key, value in cf.items("DAISY"):

            option_dict[key] = eval(value)

        # print(option_dict)
        return option_dict        
    
    #read image
    def read_image(self, image_name, size=None):
        options = self.get_options()
        
        if size:
            im = np.array(Image.open(image_name).convert("L").resize(size))
        else:
            im = np.array(Image.open(image_name).convert("L"))
        
        options["image"] = im
        feature = daisy(**options)
        return feature.reshape((1, feature.shape[0] * feature.shape[1]* feature.shape[2]))[0]


if __name__ == '__main__':
    for file in os.listdir("image"):
        im = Image.open(os.path.join("image", file)).convert("L")
        size = im.size
        feature = DAISY().read_image(os.path.join("image", file), size)
    # get_options()
