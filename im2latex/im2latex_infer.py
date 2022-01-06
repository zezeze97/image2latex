import os
import torch
from mmcv.image import imread
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401
import json

def build_model(config_file, checkpoint_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = build_model(config_file, checkpoint_file)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Specify GPU device
            device = torch.device("cuda:{}".format(device))

        self.model.to(device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass

class Recognition_Inference(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=64):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, preds, file_path=None):
        results = []
        for pred in preds:
            if len(pred['score']) == 0:
                pred['score'] = 0.
            else:
                pred['score'] = sum(pred['score']) / len(pred['score'])
            results.append(pred)
        return results

    def predict_batch(self, imgs):
        # predict one image, load batch_size crop images.
        batch = []
        all_results = []
        for i, img in enumerate(imgs):
            batch.append(img)
            if len(batch) == self.samples_per_gpu:
                results = model_inference(self.model, batch, batch_mode=True)
                all_results += results
                batch = []
        # rest length
        if len(batch) > 0:
            results = model_inference(self.model, batch, batch_mode=True)
            all_results += results
        all_results = self.result_format(all_results)
        return all_results


def read_img(img_path):
    img_name_list = os.listdir(img_path)
    img_array_list = []
    for img in img_name_list:
        img_array_list.append(imread(img_path + img))
    return img_name_list, img_array_list

def save_result(img_name_list, result, output_dir):
    pred_results = []
    for i,img_name in enumerate(img_name_list):
        pred_result = {}
        pred_result['image name'] = img_name
        pred_latex = ''
        for token in result[i]['text']:
            pred_latex += token + ' '
        pred_result['latex pred'] = pred_latex.strip(' ')
        pred_results.append(pred_result)
    with open(output_dir + 'pred.json','w')as f:
        json.dump(pred_results, f)


if __name__ =='__main__':
    config_file = 'configs/textrecog/master/master_Resnet31_withGCB_im2latex.py'
    checkpoint_file = 'expr_result/im2latex_res31/pretrained.pth'
    img_path = '/home/zhangzr/im2latex_data/test_img/'
    output_dir = 'expr_result/im2latex_res31/predict/'


    im2latex_model = Recognition_Inference(config_file, checkpoint_file, samples_per_gpu=32)
    img_name_list, img_array_list = read_img(img_path)
    result = im2latex_model.predict_batch(img_array_list)
    save_result(img_name_list, result, output_dir)