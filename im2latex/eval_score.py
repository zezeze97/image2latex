from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json

prediction_file = 'expr_result/im2latex_res31/predict/pred.json'
gt_file = '/home/zhangzr/im2latex_data/master_data/test.txt'

with open(prediction_file,'r')as f:
    pred_data = json.load(f)

with open(gt_file, 'r')as f:
    gt_data = f.readlines()

pred_dict = {}
for item in pred_data:
    pred_dict[item['image name']] = item['latex pred'].split(' ')

gt_dict = {}
for item in gt_data:
    temp_list = item.split('\t')
    img_name = temp_list[0]
    latex = temp_list[1:-1]
    gt_dict[img_name] = latex

bleu_gram1_list = []
bleu_gram2_list = []
bleu_gram3_list = []
bleu_gram4_list = []


for item in pred_dict.keys():
    pred = pred_dict[item]
    file_name = item.split('/')[-1]
    gts = gt_dict[file_name]
    bleu_gram1_list.append(sentence_bleu(gts,pred,weights=(1, 0, 0, 0)))
    bleu_gram2_list.append(sentence_bleu(gts,pred,weights=(0.5, 0.5, 0, 0)))
    bleu_gram3_list.append(sentence_bleu(gts,pred,weights=(0.33, 0.33, 0.33, 0)))
    bleu_gram4_list.append(sentence_bleu(gts,pred,weights=(0.25, 0.25, 0.25, 0.25)))
    # bleu_gram4_list.append(sentence_bleu(gts,pred))
    
print('bleu@1 score: ',np.mean(bleu_gram1_list))
print('bleu@2 score: ',np.mean(bleu_gram2_list))
print('bleu@3 score: ',np.mean(bleu_gram3_list))
print('bleu@4 score: ',np.mean(bleu_gram4_list))

