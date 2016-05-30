#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy
import collections
import simulator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle

class NCCTracker(object):

    def __init__(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[top:bottom, left:right]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))

        right = int(min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1))

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[top:bottom, left:right]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1])

