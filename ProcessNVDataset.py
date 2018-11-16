import os
import glob
import numpy as np
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import shutil
import json
import random
import cv2


boundingboxlist = []

def make_xml(boundingboxlist, File_name, W, H):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC1988'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = File_name + '.png'

    node_Source = SubElement(node_root, 'source')
    node_database = SubElement(node_Source, 'database')
    node_database.text = 'Unknown'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(W)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(H)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for i in range(len(boundingboxlist)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = boundingboxlist[i]['class']

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(boundingboxlist[i]['xmin'])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(boundingboxlist[i]['ymin'])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(boundingboxlist[i]['xmax'])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(boundingboxlist[i]['ymax'])

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    # print xml 打印查看结果
    return xml

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

if __name__ == '__main__':

    _MainDir = './data/VOCdevkit1988/VOC1988/'
    DataRoot = _MainDir + 'BottleDataSet/'

    listdir = os.listdir(DataRoot)

    TrainListNum = 5000

    FileNamelist = 0
    TrainList = []
    TestList = []

    for listDir_Temp in listdir:
        PathList = DataRoot + listDir_Temp + '/'

        Imagegroup = glob.glob(PathList + '??????.png')
        Jsongroup = glob.glob(PathList + '??????.json')

        # for Filepath in Imagegroup:
        #     TargetPath = Filepath.replace("BottleDataSet", "JPEGImages")
        #
        #     TargetPath = os.path.dirname(TargetPath)
        #     TargetPath = os.path.dirname(TargetPath)
        #     TargetPath = TargetPath + '/' + str(FileName_temp) + '.png'
        #
        #     shutil.copy2(Filepath, TargetPath)
        #     FileName_temp = FileName_temp + 1



        FileName_temp = FileNamelist

        Findclass = ['daniels', 'vodka']

        for i in range(len(Jsongroup)):
            json_file = open(Jsongroup[i])
            Data = json.load(json_file)
            FileName = os.path.basename(Jsongroup[i])

            img = cv2.imread(Imagegroup[i])
            height, width, channels = img.shape

            boundingboxlist = []
            isTrainData = 0
            for p in Data['objects']:

                strname = p['class']

                ymin = p['bounding_box']["top_left"][0]
                ymin = int(clamp(ymin, 0, height))

                xmin = p['bounding_box']["top_left"][1]
                xmin = int(clamp(xmin, 0, width))

                ymax = p['bounding_box']["bottom_right"][0]
                ymax = int(clamp(ymax, 0, height))

                xmax = p['bounding_box']["bottom_right"][1]
                xmax = int(clamp(xmax, 0, width))

                lower = strname.split('_')[-2].lower()

                HasFindClass = Findclass.count(lower)
                if len(TrainList) <= TrainListNum:
                    if isTrainData == 0:
                        isTrainData = HasFindClass

                if HasFindClass == 0:
                    continue

                boundingboxlist.append({'xmin' : xmin,'ymin' : ymin, 'xmax' : xmax,'ymax' : ymax, 'class' : lower})

            NameTemp = str(FileName_temp)
            dom = make_xml(boundingboxlist, NameTemp, width, height)
            temp = NameTemp.split('.')

            if isTrainData == 1:
                TrainList.append(temp[0])
            else:
                TestList.append(temp[0])
            xml_name = os.path.join(_MainDir + "Annotations", temp[0] + '.xml')
            with open(xml_name, "wb") as outfile:
                outfile.write(dom)

            FileName_temp = FileName_temp + 1

        FileNamelist = FileName_temp


    _ImageSets = _MainDir + "ImageSets/"
    if not os.path.exists(_ImageSets):
        os.makedirs(_ImageSets)

    _main = _ImageSets + "Main/"
    if not os.path.exists(_main):
        os.makedirs(_main)

    file = open(_main + "train.txt", "w")
    for i in TrainList:
        file.write(i + "\n")
    file.close()

    file = open(_main + "test.txt", "w")
    for i in TestList:
        file.write(i + "\n")
    file.close()


# image = cv2.imread(Imagegroup[0])
# temp = image.copy()
# for it in boundingboxlist:
#     cv2.circle(temp, (it['xmin'], it['ymin']), 5, (0, 255, 0), -1)
#     cv2.circle(temp, (it['xmax'], it['ymax']), 5, (0, 255, 0), -1)
# cv2.imshow("0", temp)