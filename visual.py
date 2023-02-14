from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
import cv2
 
def show_result_pyplot(model, img, result, score_thr=0.25):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr,bbox_color='yellow',text_color='blue', font_size=15,show=False) #cyan
    return img

 
def main():
    # config文件
    config_file = '/home/wangziqin/SSPNet-master/sspnet_result/full.py'
    # 训练好的模型
    checkpoint_file = '/home/wangziqin/SSPNet-master/sspnet_result/epoch_12.pth'
 
    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    #                 0  size=20                  1    size=40        2  size=45
    # 图片路径 bb_V0027_I0002200.jpg   youtube_V0001_I0000720.jpg   youtube_V0005_I0001760.jpg
    # name= '/home/wangziqin/mmdetection/data/coco/val2017/labeled_images/bb_V0009_I0006280.jpg'
    nn = 'baidu_P000_7'
    name= '/home/wangziqin/mmdetection/data/coco/val2017/labeled_images/{}.jpg'.format(nn)
    # 检测后存放图片路径
    out_dir = '/home/wangziqin/SSPNet-master/a_vis_labeled'
 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    result = inference_detector(model, name)
    img = show_result_pyplot(model, name, result, score_thr=0.3)
    #命名输出图片名称
    cv2.imwrite("{}/{}.jpg".format(out_dir, nn), img)
 
 
if __name__ == '__main__':
    main()