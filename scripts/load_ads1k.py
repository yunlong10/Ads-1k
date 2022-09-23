import numpy as np
import numpy as np
from torch.utils.data import Dataset

class Dataset_Ads1k(Dataset):
    def __init__(self,baseinfo,visual_features,is_train=False,ppl_maps=None,text_features=None, audio_features=None):
        super(Dataset_Ads1k,self).__init__()
        self.visual_features = visual_features
        self.baseinfo = baseinfo
        self.text_features = text_features
        self.audio_features = audio_features
        self.key=list(visual_features.keys())
        self.baseinfo_key=list(baseinfo.keys())
        self.text_features_key=list(text_features.keys())
        self.audio_features_key=list(audio_features.keys())
        self.is_train = is_train
        self.valid_key=[]
        
        MXL_f=0
        for k in self.key:
            MXL_f=max(MXL_f,len(self.visual_features[k]))

        self.inp_vec=dict()
        self.dur_vec=dict()
        self.n_fragment=dict()
        self.md5=dict()
        self.url=dict()
        self.ppl_map=dict()
        self.imps=dict()
        # print(self.key)
        for k in self.key:
            if k not in self.baseinfo_key or \
                len(self.visual_features[k])!=len(self.baseinfo[k]):
                print("视觉特征与baseinfo不对齐", k)
                continue  

            if self.text_features is not None:
                if k not in self.text_features_key:
                    print("无对应文本特征",k)
                if k not in self.text_features_key or \
                    len(self.text_features[k])!=len(self.visual_features[k]):
                    print("文本特征与视觉特征不对齐", k)
                    continue
            
            if self.audio_features is not None:
                if k not in self.audio_features_key:
                    print("无对应音频特征",k)
                if k not in self.audio_features_key or \
                    len(self.audio_features[k])!=len(self.visual_features[k]):
                    print("文本音频与视觉特征不对齐", k)
                    continue
            self.valid_key.append(k)

        for k in self.valid_key:
            if text_features is None and audio_features is None:
                    feat_dim=self.visual_features[k].shape[1]
            elif text_features is not None and audio_features is None:
                feat_dim=self.visual_features[k].shape[1]+self.text_features[k].shape[1]
            elif text_features is None and audio_features is not None:
                feat_dim=self.visual_features[k].shape[1]+self.audio_features[k].shape[1]
            else:
                feat_dim=self.visual_features[k].shape[1]+self.text_features[k].shape[1]+self.audio_features[k].shape[1]
            inp_vec=np.zeros((MXL_f,feat_dim),dtype=np.float32)
            dur_vec=np.zeros((MXL_f,),dtype=np.float32)
            ppl_map=np.zeros((MXL_f,MXL_f),dtype=np.float32)
            imp=np.zeros((MXL_f,),dtype=np.float32)

            for i in range(len(self.visual_features[k])):
                if text_features is None and audio_features is None:
                    inp_vec[i]=self.visual_features[k][i]
                elif text_features is not None and audio_features is None:
                    inp_vec[i]=np.concatenate((self.visual_features[k][i],self.text_features[k][i]),axis=0)
                elif text_features is None and audio_features is not None:
                    inp_vec[i]=np.concatenate((self.visual_features[k][i],self.audio_features[k][i]),axis=0)
                else:
                    inp_vec[i]=np.concatenate((self.visual_features[k][i],self.text_features[k][i],self.audio_features[k][i]),axis=0)
                
                if is_train:
                    imp[i]=self.baseinfo[k][i]['imp']
                    dur_vec[i]=self.baseinfo[k][i]['segment'][1]-self.baseinfo[k][i]['segment'][0]
                else:
                    dur_vec[i]=self.baseinfo[k][i][0]

            self.inp_vec[k]=inp_vec
            self.dur_vec[k]=dur_vec
            self.n_fragment[k]=len(self.baseinfo[k])
            self.md5[k]=k
            self.imps[k]=imp

            if self.is_train:
                for i in range(self.n_fragment[k]-1):
                    for j in range(i+1,self.n_fragment[k]):
                        ppl_map[i,j]=ppl_maps[k][i,j]

                self.ppl_map[k]=ppl_map
            else:
                self.url[k]=self.baseinfo[k][0][1]
                
    def __len__(self):
        return len(self.valid_key)

    def __getitem__(self,idx):
        k=self.valid_key[idx]
        while k not in self.md5.keys():
            idx+=1
            k=self.valid_key[idx]

        if self.is_train:    
            return self.inp_vec[k],self.dur_vec[k],self.n_fragment[k],self.md5[k],self.ppl_map[k],self.imps[k]
        else:
            return self.inp_vec[k],self.dur_vec[k],self.n_fragment[k],self.md5[k],self.url[k]