import numpy as np
import json
import os


def read_json(json_file):

    with open(json_file, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        json_data['filename'] = json_data['file_name'] + '.jpg'
        json_data['height'] = 540
        json_data['width'] = 960
        del (json_data['file_name'])
        lines_x = [json_data['lines'][i][:72] for i in range(len(json_data['lines']))]
        lines_y = [json_data['lines'][i][-2:] for i in range(len(json_data['lines']))]
        labels = ['crop_line'] * len(lines_x)
        json_data['ann'] = {'lines_x': lines_x, 'lines_y': lines_y, 'labels': labels}
        del(json_data['lines'])
        return json_data


def write_json(json_data, file_path):
    file_name = json_data['filename'][:-4]
    json_data = json.dumps(json_data, indent=4)
    f = open(file_path + file_name+'.json', 'w')
    f.write(json_data)
    f.close()


if __name__ == '__main__':
    label_dir = 'D:/model/laneformer/data/line/train/'
    label_list = os.listdir(label_dir)
    label_write_path = 'data/labels/train/'
    for label in label_list:
        json_file = label_dir + label
        json_data = read_json(json_file)
        write_json(json_data, label_write_path)

