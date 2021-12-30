from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json

prediction_file = 'expr_result/im2caption_chinese_transformer/predict/_2021-12-30_22-08-58/results.txt'
gt_file = '../im2caption_data/chinese_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
with open(gt_file,'r')as f:
    gt_data = json.load(f)
with open(prediction_file,'r')as f:
    pred_data = f.readlines()

gt_dict ={}
for item in gt_data:
    image_name = item['image_id']
    captions = item['caption']
    caption_list = []
    for caption in captions:
        temp = []
        for token in caption:
            temp.append(token)
        caption_list.append(temp)
    gt_dict[image_name] = caption_list

pred_dict = {}
for item in pred_data:
    item_list = item.split(' ')
    file_name = item_list[0]
    pred = item_list[1:-2]
    pred_dict[file_name] = pred

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







