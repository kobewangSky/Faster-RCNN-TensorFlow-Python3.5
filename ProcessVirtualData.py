import os
import glob
import numpy as np
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, image_name):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC1988'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name + '.jpg'

    node_Source = SubElement(node_root, 'source')
    node_database = SubElement(node_Source, 'database')
    node_database.text = 'Unknown'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '640'
    node_height = SubElement(node_size, 'height')
    node_height.text = '480'
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for i in range(1):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'vodka'

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin_tuple)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin_tuple)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax_tuple)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax_tuple)


    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    #print xml 打印查看结果
    return xml


_MainName = 'VOCdevkit1988/'
_MainDir = "./data/" + _MainName + "VOC1988/"


ImageData = np.load(_MainDir + "LableList_justPic_FOV_rang_43_HD_RealWall.npy" )

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

for i in range(len(ImageData)):

    xmin = int(clamp(ImageData[i][0], 0, 640))
    ymin = int(clamp(ImageData[i][3], 0, 480))
    xmax = int(clamp(ImageData[i][2], 0, 640))
    ymax = int(clamp(ImageData[i][1], 0, 480))

    if xmin > xmax:
        print("error")
    if ymin > ymax:
        print("error")

    if xmin > 640:
        print("error")
    if ymin > 480:
        print("error")
    if xmax > 640:
        print("error")
    if ymax > 640:
        print("error")

    dom = make_xml(xmin, ymin, xmax, ymax, "Fake" + str(i))
    xml_name = os.path.join(_MainDir + "Annotations", "Fake" + str(i) + '.xml')
    with open(xml_name, "wb") as outfile:
        outfile.write(dom)


# with open(xml_name, 'wb') as f:
#     f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))



_Txt = _MainDir + "ImageSets/Main/"

file = open(_Txt + "train.txt", "w")
for i in range(0, 9000):
    file.write("Fake" + str(i) + "\n")
file.close()

file = open(_Txt + "test.txt", "w")
for i in range(9000, 10000):
    file.write("Fake" + str(i) + "\n")
file.close()

