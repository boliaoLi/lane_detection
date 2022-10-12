from lanedet.core import eval_map, eval_recalls
import numpy as np


def creat_det_results(num_img, num_class):
    det_results = []
    for i in range(num_img):
        det_result = []
        for j in range(num_class):
            lines = np.random.rand(np.random.randint(10, 20), 75)
            det_result.append(lines)
        det_results.append(det_result)
    return det_results


def creat_annotations(num_img, num_class):
    """
    annotations (list[dict]): Ground truth annotations where each item of
    the list indicates an image. Keys of annotations are:

    - `lines`: numpy array of shape (n, 74)
    - `labels`: numpy array of shape (n, )
    - `lines_ignore` (optional): numpy array of shape (k, 74)
    - `labels_ignore` (optional): numpy array of shape (k, )
    """
    annotations = []
    for i in range(num_img):
        line_num = np.random.randint(5, 10)
        lines = np.random.rand(line_num, 74)
        labels = np.random.randint(0, num_class, (line_num, ))
        annotation = {'lines': lines, 'labels': labels}
        annotations.append(annotation)
    return annotations


if __name__ == '__main__':
    num_class = 4
    num_img = 2
    det_results = creat_det_results(num_img, num_class)
    annotations = creat_annotations(num_img, num_class)
    mean_ap, _ = eval_map(det_results, annotations)
    print(mean_ap)