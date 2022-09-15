from lanedet.core.visualization.image import imshow_gt_det_lines, imshow_det_lines
import mmcv
import numpy as np


def test_gt_det_lines():
    img = 'demo/demo_img.jpg'
    class_names = ['cat', 'dog', 'sheep', 'elephant']

    # 构造测试ground truth
    lines_x = []
    for i in range(6):
        line_x = np.linspace(np.random.randint(0, 500), np.random.randint(500, 960), 70)
        lines_x.append(line_x)
    lines_x = np.vstack(lines_x)
    # gt_lines = np.concatenate([np.random.randint(0, 960, (6, 70)), np.random.randint(0, 10, (6, 1)), np.random.randint(500, 540, (6, 1))], axis=1)
    gt_lines = np.concatenate([lines_x, np.random.randint(0, 10, (6, 1)), np.random.randint(500, 540, (6, 1))], axis=1)
    gt_labels = np.array([0, 1, 0, 3, 2, 2])
    annotations = {'gt_lines': gt_lines, 'gt_labels': gt_labels}

    # 构造测试result
    pred_lines = []
    for i in range(400, 960, 20):
        pred_line_x = np.linspace(np.random.randint(0, 400), i, 70)
        pred_line_y = np.array([np.random.randint(0, 220), np.random.randint(320, 540)])
        pred_line = np.concatenate([pred_line_x, pred_line_y])
        pred_lines.append(pred_line)
    pred_lines = np.vstack(pred_lines)
    pred_scores = np.random.random(size=pred_lines.shape[0])
    pred_labels = np.random.randint(0, 4, pred_lines.shape[0])
    result = dict(pred_lines=pred_lines,
                  pred_scores=pred_scores,
                  pred_labels=pred_labels)

    img = imshow_gt_det_lines(img=img,
                              annotation=annotations,
                              result=result,
                              class_names=class_names
                              )
    # img = imshow_det_lines(img,
    #                        pred_lines,
    #                        pred_labels)
    # mmcv.imshow(img)


if __name__ == '__main__':
    test_gt_det_lines()
