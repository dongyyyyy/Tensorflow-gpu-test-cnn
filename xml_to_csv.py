import os
import glob
import cv2
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + './*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            img = cv2.imread(path + root.find('filename').text, cv2.IMREAD_ANYCOLOR)
            height, width, channel = img.shape
            value = (root.find('filename').text,
                     width, # image Width
                     height, # image Height
                     member[0].text, #
                     float(int(member[4][0].text))/width,
                     float(int(member[4][1].text))/height,
                     float(int(member[4][2].text))/width,
                     float(int(member[4][3].text))/height
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(),'dataset/{}/'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('dataset/{}_labels.csv'.format(directory),index=None)
        print("Successfully convert xml to csv")

main()