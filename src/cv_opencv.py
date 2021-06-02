import cv2
import numpy as np

class RGB_Cam(object):
    color_stream = cv2.VideoCapture(0)

    @classmethod
    def read(cls):
        return None, cls.color_stream.read()[1]