#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from voc_eval import voc_eval
import os
import glob

from typing import Dict, List

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
# CLASSES = ('__background__',  # always index 0
#                          'daniels',
#                          'vodka',
#                          'sapphire',
#                          'line001',
#                          'line003',
#                          'campari',
#                          'gin',
#                          'club',
#                          'b',
#                          'cuervo',
#                          'bianco',
#                          'sambuca',
#                          'sauza')

CLASSES = ('__background__',  # always index 0
                         'daniels',
                         'vodka',)

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_80000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'voc_1988_train': ('voc_1988_train',)}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] > thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs = [], [], [], []

ndraw = 0

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    #im = cv2.imread(im_file)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.1



    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        basename = os.path.basename(image_name)
        temp = basename.split('.')[0]

        all_image_ids.extend([temp] * len(dets))
        all_detection_bboxes.extend(dets[:,0:4].tolist())
        all_detection_probs.extend(dets[:, 4].tolist())
        all_detection_labels.extend([cls] * len(dets))

        #vis_detections(im, cls, dets, thresh=CONF_THRESH)



def _write_results( image_ids, bboxes, labels, probs):
    label_to_txt_files_dict = {}
    for c in range(1, len(CLASSES)):
        label_to_txt_files_dict[CLASSES[c]] = open(os.path.join('C:/Users/kobe/Faster-RCNN-TensorFlow-Python3.5/data', 'comp3_det_test_{:s}.txt'.format(
            CLASSES[c])), 'w')

    for image_id, bbox, label, prob in zip(image_ids, bboxes, labels, probs):
        label_to_txt_files_dict[label].write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(image_id, prob,
                                                                                      bbox[0], bbox[1], bbox[2],
                                                                                      bbox[3]))

    for _, f in label_to_txt_files_dict.items():
        f.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 voc_1988_train]',
                        choices=DATASETS.keys(), default='voc_1988_train')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                            tag='default')
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    Testfile = "./data/VOCdevkit1988/VOC1988/ImageSets/Main/test.txt"
    file = open(Testfile, "r")
    TestList = file.readlines()
    file.close()
    im_names = []
    for i in range(len(TestList)):
        im_names.append( './data/VOCdevkit1988/VOC1988/JPEGImages/' + TestList[i].split('\n')[0] + '.jpg')

    print('Loaded network {:s}'.format(tfmodel))


    # im_names = [ '4dvi5rfsk5c01.jpg',
    #             '2-havana-club-anejo-blanco-reserva-rum-bottle_1_cd08917584dea27261b2021775947a6b.jpg',
    #             '28709b34-bbd0-4e7f-bf51-e89f75b55740.jpg',
    #             'be1c54a2-203c-465a-a62b-1ba249b367ee.jpg',
    #             'campari-milano-italy-(750ml)-400px-400px.jpg',
    #             'fJ3J1YC.jpg',
    #             'images (1).jpg',
    #             'images (2).jpg',
    #             'images.jpg',
    #             'jd.jpg',
    #             'WS_Jack+Daniels.jpg',
    #             'test1.png',
    #             'test2.png',
    #             'test3.png',
    #             'test4.png',
    #             '99033_absolut_1l_edo_3.jpg',
    #             'Absolut.jpg',
    #             'absolut_1024x1024.jpg',
    #             'absolut-2.jpg',
    #             'absolut----408-1b5886c.jpg',
    #             'Absolut-Vodka.jpg',
    #             'absolut-vodka-45-l.jpg',
    #             'ABSOLUT-VODKA-200ML.jpg',
    #             'ABSOLUT-VODKA-375ML.jpg',
    #             'c7e221fc72ede293318bb6de9e213ec3.jpg',
    #             'img_4208_2-126c837.jpg']
    # for i in range(len(im_names)) :
    #     im_names[i] = './data/demo/' + im_names[i]

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    #plt.show()


    _write_results(all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs)

    path_to_voc2007_dir = os.path.join('C:\\Users\\kobe\\Faster-RCNN-TensorFlow-Python3.5\\data', 'VOCdevkit1988',
                                       'VOC1988')
    path_to_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
    path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')


    label_to_ap_dict = {}
    for c in range(1, len(CLASSES) ):
        category = CLASSES[c]
        try:
            _, _, ap, overlap = voc_eval(
                detpath=os.path.join('C:/Users/kobe/Faster-RCNN-TensorFlow-Python3.5/data', 'comp3_det_test_{:s}.txt'.format(category)),
                annopath=os.path.join(path_to_annotations_dir, '{}.xml'),
                imagesetfile=os.path.join(path_to_main_dir, 'test.txt'),
                classname=category,
                cachedir='cache',
                ovthresh=0.5,
                use_07_metric=True)
        except IndexError:
            ap = 0

        label_to_ap_dict[c] = ap

        print("predict " + category + " = " + str(ap) )
    #plt.show()