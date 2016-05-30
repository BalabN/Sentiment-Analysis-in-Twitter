import cv2
import numpy as np
from numpy import *
import vot
import matplotlib.pyplot as plt

class flow(object):
    def __init__(self, image, region):

        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[top:bottom, left:right]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        self.old_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(image)
        self.hsv[..., 1] = 255


    def set_region(self, position):
        self.position = position


    def track(self, image):
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = int(min(round(self.position[0] + float(self.window) / 2), image2.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image2.shape[0] - 1))

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image2[top:bottom, left:right]
        cut_prev = self.old_img[top:bottom, left:right]
        flow2 = cv2.calcOpticalFlowFarneback(cut_prev, cut, 0.5, 1, 5, 15, 3, 2, 1)

        self.old_img = image2
        npflowx = np.array(flow2[..., 0])[self.position[1]-top:self.position[1]-top + self.size[1], self.position[0]-left:self.position[0]-left + self.size[0]]
        npflowy = np.array(flow2[..., 1])[self.position[1]-top:self.position[1]-top + self.size[1], self.position[0]-left:self.position[0]-left + self.size[0]]

        mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

        self.hsv = np.zeros_like(image[top:bottom, left:right, :])
        self.hsv[..., 1] = 255
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2RGB)

        #matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        #image[top:bottom, left:right] = rgb
        #print([self.position[1]-top,self.position[1]-top + self.size[1], self.position[0]-left,self.position[0]-left + self.size[0]])
        i = 0
        sumx = 0
        sumy = 0
        while(i < 1):
            meanx = np.mean(npflowx)
            meany = np.mean(npflowy)
            #print(meany)
            sumx += meanx
            sumy += meany
            npflowx -= meanx
            npflowy -= meany
            i+=1
        #print(sumx)
        #print(sumy)
        self.position = (int(self.position[0]+sumx), int(self.position[1]+sumy))
        #a = plt.imshow(image)
        return vot.Rectangle(left+sumx+self.size[0]/2, top+sumy+self.size[1]/2, self.size[0], self.size[1])
