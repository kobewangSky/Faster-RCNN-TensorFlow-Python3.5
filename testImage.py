import cv2
import glob
import xml.etree.ElementTree as ET


_MainName = 'VOCdevkit1988/'
_MainDir = "./data/" + _MainName + "VOC1988_RealData/"

Imagegroup = sorted(glob.glob(_MainDir + "JPEGImages/" + '*.jpg'))
xmlgroup = sorted(glob.glob(_MainDir + "Annotations/" + '*.xml'))

for i in range(0, len(Imagegroup)):
    image = cv2.imread(Imagegroup[i])
    image = image.copy()

    doc = ET.parse(xmlgroup[i])
    root = doc.getroot()
    xmin = doc.findall('object/bndbox/xmin')
    ymin = doc.findall('object/bndbox/ymin')
    xmax = doc.findall('object/bndbox/xmax')
    ymax = doc.findall('object/bndbox/ymax')




    for j in range(0 , len(xmin)):
        xmin_temp = (int(xmin[j].text))
        ymin_temp = (int(ymin[j].text))
        xmax_temp = (int(xmax[j].text))
        ymax_temp = (int(ymax[j].text))

        cv2.circle(image, (xmin_temp, ymin_temp), 10, (0, 255, 0), -1)
        cv2.circle(image, (xmax_temp, ymax_temp), 10, (0, 0, 255), -1)
        print(i)
    cv2.imshow("0", image)
    cv2.waitKey()
        # if (xmin_temp == 0  and xmax_temp == 512):
        #     image = cv2.resize(image, (640, 480))
        #     cv2.imshow("0", image)
        #     cv2.waitKey()

    # image = cv2.resize(image, (640, 480))
    # cv2.imshow("0", image)
    # cv2.waitKey()
