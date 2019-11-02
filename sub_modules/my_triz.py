import os
from sub_modules import TriZ
from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigvals
from PIL import Image

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

import numpy as np


#from matplotlib import pyplot as plt


class TRIZ(object):
    """ The PEAK module
    """
    def __str__(self):
        return "\nUsing the algorithm TRIZ.....\n"

    def get_name(self):
        return "TRIZ"       

    # read the configure file    
    def get_options(self):
        cf = ConfigParser.ConfigParser()
        cf.read("config.cof")

        option_dict = dict()

        for key, value in cf.items("STRUCTURE"):

            option_dict[key] = eval(value)

        # print(option_dict)
        return option_dict

    # normalize the features    
    def normalize(self, feature):

        normalizer = MinMaxScaler()
        normalized_feature = normalizer.fit_transform(feature)

        return normalized_feature

    # read image    
    def read_image(self, image_name, size=None):
        options = self.get_options()
        all_feature = []
        if size:
            im = np.array(Image.open(image_name).convert("L").resize(size))
        else:
            im = np.array(Image.open(image_name).convert("L"))

        options["image"] = im
        magnitude,direction = TriZ.get_magnitude(im)

        all_start_point,chunk_size = TriZ.get_all_start_point(magnitude,scale_factor = 5)  #[[(20, 35)], [(16, 27), (20, 27), (24, 27), (24, 35), (16, 35), (16, 43), (20, 43), (24, 43)], [(12, 19), (16, 19),......
        raw_feature_list = TriZ.get_chunk_feature(all_start_point,chunk_size,magnitude,direction)
        tmp_feature = TriZ.transform_feature(raw_feature_list, k= 9)
        all_feature.append(tmp_feature)
        feature = np.array(all_feature)
        # plt.imshow(feature)
        # plt.show()
        return feature.reshape((1, feature.shape[0] * feature.shape[1]))[0]

if __name__ == '__main__':
    feature = STRUCTURE().read_image("../img_SUB/Gastric_polyp_sub/Erosionscromatosc_1_s.jpg")
    print(feature)
    # get_options()
