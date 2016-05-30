import cv2

from os import listdir
from os.path import isfile, join
import cv2.cv as cv
from time import time
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
class simulator():
    def __init__(self, path):
        self.onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        self.regionk = []
        self.image = []
        self.stevec = 1
        self.boxes = []
        self.img = []
        self.path = path
        self.add_region()




    def frame(self):
        for url in self.onlyfiles:
            if int('%08d' % int(url[:-4])) == self.stevec:
                self.stevec += 1
                return self.path+url
        return None

    def report(self, regionk):
        self.regionk = regionk
        #print("plot")





    def region(self):
        return self.regionk




    def on_mouse(self, event, x, y, flags, params):
        # global img
        t = time()
        if len(self.boxes) > 0:
            cv2.rectangle(self.img, (self.boxes[-1][0], self.boxes[-1][1]), (x, y), (0, 255, 0))
        if event == cv.CV_EVENT_LBUTTONDOWN:
             print 'Start Mouse Position: '+str(x)+', '+str(y)
             sbox = [x, y]
             self.boxes.append(sbox)


        elif event == cv.CV_EVENT_LBUTTONUP:
            print 'End Mouse Position: '+str(x)+', '+str(y)
            ebox = [x, y]
            self.boxes.append(ebox)
            print self.boxes
            crop = self.img[self.boxes[-2][1]:self.boxes[-1][1],self.boxes[-2][0]:self.boxes[-1][0]]
            self.regionk = Rectangle(self.boxes[-2][0], self.boxes[-2][1], self.boxes[-1][0]-self.boxes[-2][0], self.boxes[-1][1]-self.boxes[-2][1])
            print(self.regionk)
            self.boxes = []



    def add_region(self):
        count = 0
        count += 1
        url =  self.frame()
        self.img = cv2.imread(url)
        cv2.namedWindow('real image')
        cv.SetMouseCallback('real image', self.on_mouse, 0)
        cv2.startWindowThread()
        while(1):
            cv2.imshow('real image', self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
        print("naprej")

