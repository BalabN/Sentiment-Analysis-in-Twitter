import cv2
import numpy as np
from numpy import *
import vot
import random
import matplotlib.pyplot as plt
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal
from mondrianforest import process_command_line, MondrianForest


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class flow(object):
    def __init__(self, image, region):

        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        if (right - left) % 2 != 0:
            right -= 1
        if (bottom - top) % 2 != 0:
            bottom -= 1

        self.template = image[top:bottom, left:right]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)
        self.old_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(image)
        self.hsv[..., 1] = 255

        self.pos = np.array([image[top:bottom, left:right].copy().tolist()])
        self.neg = np.array([])

        vred = [(0, 0)]
        while (1):
            l = random.randint(-int((right - left) * 0.1), int((right - left) * 0.1))
            t = random.randint(-int((bottom - top) * 0.1), int((bottom - top) * 0.1))
            if l + left >= 0 and l + right < image.shape[1] and t + top >= 0 and t + bottom < image.shape[0]:
                vred += [(l, t)]
            if len(vred) > 5:
                break
        self.pos = np.array(
            [np.array(self.old_img[top + t:bottom + t, left + l:right + l].copy().tolist()) for (l, t) in vred])
        vred = []
        infloop = 0
        while (1):
            l = random.randint((right - left) / 2, image.shape[1] - (right - left) / 2 - 1)
            t = random.randint((bottom - top) / 2, image.shape[0] - (bottom - top) / 2 - 1)
            if abs(l - left - (right - left) / 2) > (right - left) and abs(t - top - (bottom - top) / 2) > (
                        bottom - top):
                vred += [(l, t)]
            if len(vred) > 5 or infloop > 10000:
                break
            infloop+=1
        self.neg = np.array(
            [np.array(self.old_img[t - (bottom - top) / 2:t + (bottom - top) / 2,
                      l - (right - left) / 2:l + (right - left) / 2].copy().tolist()) for (l, t) in vred])

        print("tu je image")
        stevec = 1
        for i in self.pos:
            cv2.imwrite("file" + str(stevec) + ".png", i)
            stevec += 1


        # Resetting random seed
        set = {'optype': 'class', 'verbose': 1, 'draw_mondrian': 0, 'perf_dataset_keys': ['train', 'test'],
               'data_path': '../../process_data/', 'dataset': 'toy-mf', 'tag': '', 'alpha': 0, 'bagging': 0,
               'select_features': 0, 'smooth_hierarchically': 1, 'normalize_features': 1, 'min_samples_split': 2,
               'save': 0, 'discount_factor': 10, 'op_dir': 'results', 'init_id': 1, 'store_every': 0,
               'perf_store_keys': ['pred_prob'], 'perf_metrics_keys': ['log_prob', 'acc'], 'budget': -1.0,
               'n_mondrians': 10, 'debug': 0, 'n_minibatches': 2, 'name_metric': 'acc', 'budget_to_use': inf}
        self.settings = Map(set)
        reset_random_seed(self.settings)
        stevec = -30
        for i in self.neg:
            cv2.imwrite("file" + str(stevec) + ".png", i)
            stevec -= 1
        x_trainp = [x.flatten().tolist() for x in self.pos]
        x_trainn = [x.flatten().tolist() for x in self.neg]
        x_train = x_trainp + x_trainn
        print(len(x_train[0]))
        self.data = {'n_dim': 1, 'x_test':
                array([x_train[5]]),
                'x_train': array(x_train),
                'y_train': array(np.ones(len(self.pos)).astype(int).tolist() + np.zeros(len(self.neg)).astype(int).tolist()), 'is_sparse': False, 'n_train': len(x_train), 'n_class': 2,
                'y_test': array([]),
                'n_test': 0}

        self.param, self.cache = precompute_minimal(self.data, self.settings)

        self.mf = MondrianForest(self.settings, self.data)

        for idx_minibatch in range(self.settings.n_minibatches):
            #train_ids_current_minibatch = self.data['train_ids_partition']['current'][idx_minibatch]
            if idx_minibatch == 0:
                # Batch training for first minibatch
                self.mf.fit(self.data, array(range(0, len(x_train)/2)), self.settings, self.param, self.cache)
            else:
                # Online update
                self.mf.partial_fit(self.data, array(range(len(x_train)/2, len(x_train))), self.settings, self.param, self.cache)
                print("updatalo je")

        weights_prediction = np.ones(self.settings.n_mondrians) * 1.0 / self.settings.n_mondrians
        #train_ids_cumulative = self.data['train_ids_partition']['cumulative'][idx_minibatch]
        print(self.mf.evaluate_predictions(self.data, array([x_train[5]]), [1], \
                    self.settings, self.param, weights_prediction, False))



    def set_region(self, position):
        self.position = position



    def updateTree(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        t = len(self.template) / 2
        l = len(self.template[0]) / 2
        left = self.position[0] -l
        top = self.position[1] -t
        right = self.position[0] +l
        bottom = self.position[1] +t

        #self.pos = np.array([image[top:bottom, left:right].copy().tolist()])
        #self.neg = np.array([])
        print(t)
        print(l)
        vred = [(0, 0)]
        while (1):
            l2 = random.randint(-int((right - left) * 0.1), int((right - left) * 0.1))
            t2 = random.randint(-int((bottom - top) * 0.1), int((bottom - top) * 0.1))
            if l2 + left >= 0 and l2 + right < image.shape[1] and t2 + top >= 0 and t + bottom < image.shape[0]:
                vred += [(l2, t2)]
            if len(vred) > 5:
                break
        self.pos = np.array(
            [np.array(image[top + t2:bottom + t2, left + l2:right + l2].copy().tolist()) for (l2, t2) in vred])
        vred = []

        infloop = 0


        while (1):
            l2 = random.randint((right - left) / 2, image.shape[1] - (right - left) / 2 - 1)
            t2 = random.randint((bottom - top) / 2, image.shape[0] - (bottom - top) / 2 - 1)
            if abs(l2 - left - (right - left) / 2) > (right - left) and abs(t2 - top - (bottom - top) / 2) > (
                        bottom - top):
                vred += [(l2, t2)]
            if len(vred) > 5 or infloop > 10000:
                break
            infloop+=1
        print(infloop)
        self.neg = np.array(
            [np.array(image[t2 - (bottom - top) / 2:t2 + (bottom - top) / 2,
                    l2 - (right - left) / 2:l2 + (right - left) / 2].tolist()) for (l2, t2) in vred])


        stevec = -1

        x_trainp = [x.flatten().tolist() for x in self.pos]
        x_trainn = [x.flatten().tolist() for x in self.neg]
        x_train = x_trainp + x_trainn
        self.data['x_train'] = np.append(self.data['x_train'], array(x_train), axis=0)
        self.data['y_train'] = np.append(self.data['y_train'], array(np.ones(len(self.pos)).astype(int).tolist() + np.zeros(len(self.neg)).astype(int).tolist()))
        #self.param, self.cache = precompute_minimal(self.data, self.settings)
        self.mf.partial_fit(self.data, array(range(len(self.data['x_train'])-len(x_train), len(self.data['x_train']))), self.settings, self.param, self.cache)



    def track(self, image):
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = int(min(round(self.position[0] + float(self.window) / 2), image2.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image2.shape[0] - 1))

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                                 self.size[1])

        cut = image2[top:bottom, left:right]

        t = len(self.template)/2
        l = len(self.template[0])/2
        weights_prediction = np.ones(self.settings.n_mondrians) * 1.0 / self.settings.n_mondrians
        predicta = []
        for i in range(t, bottom-top-t, 15):
            for j in range(l, right-left-l, 15):
                imclass = cut[i-t:i+t, j-l:j+l]
                pred = self.mf.evaluate_predictions(self.data, array([imclass.flatten().tolist()]), [1], \
                                                   self.settings, self.param, weights_prediction, False)[0]
                #print(pred['pred_prob'][0])
                predicta += [(pred['pred_prob'][0][1],i,j)]

        terk = predicta[0]
        for i in range(0, len(predicta)):
            if terk[0] < predicta[i][0]:
                terk = predicta[i]

        print(terk)


        # matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        # image[top:bottom, left:right] = rgb
        self.position = (left + terk[2], top + terk[1])
        self.updateTree(image)
        # a = plt.imshow(rgb)
        return vot.Rectangle(left+ terk[2]-l, top + terk[1] -t, self.size[0], self.size[1])
