import os
import glob
import numpy as np
import xml.etree.ElementTree as ET



if __name__ == '__main__':

    maindir = "VOC1988/"
    DataList = []

    _dir = "./data/VOCDevkit1988/" + maindir + "Annotations/"

    file_name = os.listdir(_dir)

    for xml in file_name:

        DataList.append(xml)

        doc = ET.parse(_dir + xml)
        root = doc.getroot()
        elems = root.findall("folder")
        for elem in elems:
            elem.text = maindir
        doc.write(_dir + xml)


    _ImageSets = "./data/VOCDevkit1988/" + maindir + "ImageSets/"
    if not os.path.exists(_ImageSets):
        os.makedirs(_ImageSets)

    _main = _ImageSets + "Main/"
    if not os.path.exists(_main):
        os.makedirs(_main)


    file = open(_main + "test.txt", "w")
    for i in range(0 , 5):
        temp = os.path.splitext(DataList[i])
        file.write(temp[0] + "\n")
    file.close()

    file = open(_main + "train.txt", "w")
    for i in range(5, len(DataList)):
        temp = os.path.splitext(DataList[i])
        file.write(temp[0] + "\n")
    file.close()
