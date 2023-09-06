from __future__ import division, print_function, absolute_import

import os
#import utils.cuda_util0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import time

from file_helper import write, safe_remove


def extract_info(dir_path):
    infos = []
    
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))
       
        
    return infos






def sort_similarity(result):
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort


def map_rank_quick_eval(query_info, test_info, result_argsort):
    # much more faster than hehefan's evaluation
    match = []
    junk = []
    QUERY_NUM = len(query_info)

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(test_info)):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)
    ##===== added by me 
    #print(junk)

    rank_1 = 0.0
    rank_5 = 0.0
    rank_10 = 0.0
    mAP = 0.0
    for idx in range(len(query_info)):
        #if idx % 100 == 0:
        #    print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        #============ 
        #print('idx=%d' %idx)
        #print(IGNORE)
        #============

        ig_cnt = 0
        for ig in IGNORE:
            if len(YES) > 0 and ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if len(YES) > 0 and ig_cnt >= YES[0]:
            rank_1 += 1
        if len(YES) > 0 and ig_cnt >= YES[0] - 4:
            rank_5 += 1
        if len(YES) > 0 and ig_cnt >= YES[0] - 9:
            rank_10 += 1
        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    rank5_acc = rank_5 / QUERY_NUM
    rank10_acc = rank_10 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('Rank 5:\t%f' % (rank_5 / QUERY_NUM))
    print('Rank 10:\t%f' % (rank_10 / QUERY_NUM))
    print('mAP:\t%f' % mAP)
    # np.savetxt('rank_1.log', np.array(rank1_list), fmt='%d')
    return rank1_acc, rank5_acc, rank10_acc, mAP

