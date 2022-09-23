import pickle
import json
import time
import os
import math

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from load_ads1k import Dataset_Ads1k


class Eval():
    def __init__(self):
        super(Eval, self).__init__()
        self.info_dir = os.path.join('..', 'data', 'test_info.json')
        self.seg_labels_dir = os.path.join('..', 'data', 'seg_labels_test.json')
        self.lianguan_labels_dir = os.path.join('..', 'data', 'coh_anno_test.json')

        self.visual_features_dir = os.path.join('..', 'data', 'swin_feats_test.pkl')
        self.text_features_dir = os.path.join('..', 'data', 'bert_feats_test.pkl')
        self.audio_features_dir = os.path.join('..', 'data','vggish_feats_test.pkl')

        self.visual_features = pickle.load(open(self.visual_features_dir, 'rb'))
        self.text_features = pickle.load(open(self.text_features_dir, 'rb'))
        self.audio_features = pickle.load(open(self.audio_features_dir,'rb'))
        self.baseinfo = json.load(open(self.info_dir, 'rb'))

        self.seg_info = json.load(open(self.seg_labels_dir, 'rb'))
        self.lianguan_label =  json.load(open(self.lianguan_labels_dir, 'rb'))
        self.ld  =  DataLoader(Dataset_Ads1k(
                                                baseinfo = self.baseinfo,
                                                visual_features = self.visual_features,
                                                is_train = False,
                                                text_features = self.text_features,
                                                audio_features = self.audio_features
                                            ),
                                            batch_size = 1,
                                            shuffle = False)
       
        self.imp_labels = [[
                "平铺直叙_其他",
                "对比反差_其他",
                "疑问悬念_其他",
                "情感共鸣_其他",
                "背景铺垫_其他"
            ],
            [
                "自我陈述",
                "名人介绍",
                "发布通知",
                "对话引入",
                "旁白陈述",
                "商品引入",
                "事实描述",
                "人物纵向对比",
                "人物横向对比",
                "行为对比",
                "商品/产品对比",
                "设问",
                "冲突质问",
                "元素猎奇",
                "剧情期待",
                "音乐期待",
                "道歉",
                "woops",
                "目标人群",
                "还原真实",
                "描述向往",
                "营造焦虑",
                "承诺担保",
                "重复强调",
                "精彩前置",
                "需求/痛点_其他"
            ],
            [
                "疑问反问类",
                "针对目标人群",
                "针对行为特征",
                "针对年龄阶段",
                "针对应用场景",
                "需求描述手法_其他",
                "身份需求",
                "理财收益",
                "资金缺乏",
                "时节需求",
                "生活需求",
                "娱乐需求",
                "健康需求",
                "人际关系需求",
                "工作需求",
                "学习需求",
                "保养需求",
                "需求类型_其他"
            ],
            [
                "商品功能展示",
                "商品质量展示",
                "商品整体展示",
                "商品细节展示",
                "商品使用展示",
                "模特出镜展示",
                "商品优势展示",
                "商品展示_其他",
                "环境展示",
                "业务服务展示",
                "业务效果展示",
                "业务优惠展示",
                "业务流程展示",
                "业务优势展示",
                "业务展示_其他",
                "阅读画面",
                "短视频app",
                "直播画面",
                "提现画面",
                "操作指引",
                "商品展示",
                "APP展示_其他",
                "战斗玩法",
                "趣味玩法",
                "社交玩法",
                "画风展示",
                "角色展示",
                "卖点展示",
                "装备展示",
                "玩法展示",
                "武器展示",
                "游戏提现",
                "游戏展示_其他",
                "内容展示",
                "科学认证",
                "权威背书",
                "资质展示_其他",
                "产品展示_其他",
                "品牌名称强化",
                "logo强化",
                "slogan强化",
                "公司强化",
                "IP强化",
                "代言人",
                "资质强化",
                "品牌强化_其他",
                "下载指引",
                "购买指引",
                "跳转指引",
                "行为指引_其他",
                "红包领取",
                "价格优惠",
                "虚拟礼包",
                "充值优惠",
                "新人优惠",
                "免费赠送",
                "优惠引导_其他",
                "交互指引",
                "按钮引导",
                "微动引导",
                "呼吁引导",
                "点击引导_其他",
                "行动指引_其他"
            ]]

    def score_adsprop(self, md5, result):
        filename = md5 + ".mp4"
        if filename not in self.seg_info.keys():
            return 0

        segs = self.seg_info[filename]["shapeData"]
        adsprop = 0
        n_all = 0
        for i in range(len(segs)):
            avg_seg = 0
            if result[i] == 1:
                n_all += 1
                len_labels = len(segs[i]["labels"])
                for j in segs[i]["labels"]:
                    for k in range(4):
                        if j in self.imp_labels[k]:
                            avg_seg += (0.25*(k + 1))**4

                avg_seg = avg_seg / len_labels if len_labels > 0 else 0

            adsprop += avg_seg
        return adsprop/n_all if n_all > 0 else 0

    def score_cor(self, md5, result):
        filename = md5 + ".mp4"
        if filename not in self.seg_info.keys():
            return 0

        segs = self.seg_info[filename]["shapeData"]
        lianguan0 = self.lianguan_label[md5]["连贯"]
        lianguan2 = self.lianguan_label[md5]["不确定"]
        cor = 0
        for i in range(len(segs)-1):
            for j in range(i + 1, len(segs)):
                if result[i] == 1 and result[j] == 1:
                    si = segs[i]["segment"][0]
                    sj = segs[j]["segment"][0]
                    seg_pair = []
                    seg_pair.append(si)
                    seg_pair.append(sj)

                    if seg_pair in lianguan0:
                        cor += 1
                    if seg_pair in lianguan2:
                        cor += 0.5
                    break

        if sum(result) == 1:
            return cor

        cor /= (sum(result)-1)
        return cor

    def score_time(self, result, dur_vec, given_time):

        tot_time = 0
        for i in range(len(dur_vec)):
            if result[i] == 1:
                tot_time += dur_vec[i]

        if tot_time <= given_time[1] and tot_time >= given_time[0]:
            return 1
        else:
            return 0

    def score_total(self, md5, result, dur_vec, given_time):
        return self.score_adsprop(md5, result)*self.score_cor(md5, result)*self.score_time(result, dur_vec, given_time)

    def infer(self, results, given_time):
        avg_ads_prop_score = 0
        avg_cor = 0
        avg_time = 0
        avg_score = 0
        cnt = 0

        for inp_vec, dur_vec, n_fragment, md5, _ in self.ld:
            for i in range(len(md5)):
                if sum(dur_vec[i]) < given_time[0]:
                    continue
                cnt += 1
                tmp_time_score = self.score_time(results[i], dur_vec[i], given_time)
                avg_time += tmp_time_score
                avg_ads_prop_score += self.score_adsprop(md5[i], results[i]) * tmp_time_score
                avg_cor += self.score_cor(md5[i], results[i]) * tmp_time_score
                avg_score += self.score_total(md5[i], results[i], dur_vec[i], given_time)

        if cnt ==0:
            cnt += 1
        avg_ads_prop_score = round(avg_ads_prop_score / cnt, 4)
        avg_cor = round(avg_cor / cnt, 4)
        avg_time = round(avg_time / cnt, 4)
        avg_score = round(avg_score / cnt, 4)
        
        print("|%12.8s|%14.2d|%10.4f|%10.4f|%12.4f|%14.4f|" % 
            (str(given_time), cnt, avg_ads_prop_score, avg_cor ,avg_time, avg_score))
        print('--------------------------------------------------------------------------------------------')