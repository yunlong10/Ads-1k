## Ads-1k: Dataset for Ad Video Editing 
### Dataset Overview

To obtain better training and evaluation, we collect 1000+ ad videos from the advertisers to form the Ads-1k dataset. There are 942 ad videos for training and 99 for evaluation in total. However, the annotation methods of the training set and test set are somehow different. Instead of preparing the ground-truth for each data, we annotate each video with multi-labels shown in the supplementary.

Dataset statistics. $N_{seg}$ and $N_{label}$ are respectively the average number of segments and labels of each video. $D_{seg}$ and $D_{video}$ are the average duration of a segment and a video, respectively.

|              | $N_{seg}$ | $D_{seg}$ | $D_{video}$ | $N_{label}$ |
| ------------ | --------- | --------- | ----------- | ----------- |
| Training Set | 13.90     | 2.77      | 34.60       | 30.18       |
| Test Set     | 18.81     | 1.88      | 34.21       | 35.77       |
| Overall      | 14.37     | 2.68      | 35.17       | 30.71       |

Besides, the number of annotated segment pairs and the proportion are counted. The number of *coherent*, *incoherent*, and *uncertain* pairs are 6988, 9551, and 2971, occupying 36%, 49%, and 15%, respectively.

**Note**: The ads are collected from different Chinese advertisers, both large and small. Most of them are in Chinese.

### Evaluation Metrics

#### Imp@T

$$
Imp@T=\frac{1}{|A|}\sum_{a_i\in A}imp(a_i)\cdot\mathbb{I}[c_1\cdot T\leq\tau(A)\leq c_2\cdot T]~,
$$

$$
\\
imp(a_i)=\frac{1}{|L_{a_i}|}\sum_{\ell\in L_{a_i}}w_{l}\cdot \ell~,
$$

where A is the set of selected segments, $L_{a_i}$ stands for the total number of labels of the selected segment, $\ell\in\{1,2,3,4\}$ is narrative techniques label hierarchy, and $w_{l}$ is the weight of one label. $\mathbb{I}(\cdot)$ is the indicator function. $c_1$ and $c_2$ are two constant that produced a interval based on given target duration $T$. We set $c_1=0.8$ and $c_2=1.2$, since a post-processing of $0.8\times$ slow down $1.2\times$ or fast forward can resize the result close to the target $T$ without distortion in practice. Take $T=10$ as example, the interval will be $[8,12]$, which means if the duration of result $\tau(A)=\sum_{a_i\in A}dur(a_i)\in [8,12]$ then this result is valid and gain the score.

#### Coh@T

The coherence score given the target duration $T$ is defined as follows:
$$
Coh@T=\frac{1}{|A|-1}{\sum_{a_i\prec a_j}coh{(a_{i},a_{j})}}\cdot\mathbb{I}[c_1\cdot T\leq\tau(A)\leq c_2\cdot T]~,
$$
where the $coh{(a_{i},a_{j})}$ is the coherence small score between the text of segment $i$ and the text of segment $j$. With the annotation for coherence on test set, we can score the results produced by our models. When scoring, for each combination of consecutive two segments $i$ and $j$ in output, if it is in the *coherent* set, the $coh{(a_{i},a_{j})}$ will be 1. If it is in the *incoherent* set, the $coh{(a_{i},a_{j})}$ will be 0. Otherwise, it is in the *uncertain* set, the $coh{(a_{i},a_{j})}$ will be 0.5.

#### Imp-Coh@T

The overall score is defined as follows:
$$
ImpCoh@T=\frac{Imp@T}{|A|-1}\cdot{\sum_{a_i\prec a_j}coh{(a_{i},a_{j})}}.
$$
The score reflects the ability of trade-off among importance, coherence and total duration.

### Data

We have the following dataset files under the `data` directory:

- `coh_anno_test.json`: The annotation file of coherence for 99 data in test set.
- `data_info.json`: The information of 942 ad videos data for training.
- `seg_labels_test.json`: The segments with narrative techniques labels of each video in test set.
- `test_info.json`: The information of 99 ad videos data for testing

- `bert_feats_test.pkl`:  The features of text contents (subtitles) extracted by BERT.
- `swin_feats_test.pkl`:  The features of visual infomation  (frames) extracted by Swin-Transformer (Large) from videos for test.
- `vggish_feats_test.pkl`: The features of audios extracted by Vggish from videos for test.

We also have the following pre-extracted segment-level features of training data, which can be downloaded from [[Google Drive](https://drive.google.com/file/d/1LTOCoQ_bg4hrq7IxHUvgyi-xb05e1uuR/view?usp=sharing)] or [[百度网盘](https://pan.baidu.com/s/1n5oLiFerLE-DbK-H4__T1Q?pwd=8gjb )(提取码：8gjb)]:

- `bert_feats_train.pkl`:  The features of text contents (subtitles) extracted by BERT from videos for training.
- `swin_feats_train.pkl`:  The features of visual infomation  (frames) extracted by Swin-Transformer (Large) from videos for training.
- `vggish_feats_train.pkl`: The features of audios extracted by Vggish from videos for training.
- `ppl_maps.pkl`: The PPL map of training data.

### Evaluation

Under the `scripts` directory, we include:

- `eval.py`: The evaluation script. Run `test.py` to use it.

- `load_ads1k.py`: The data loader for Ads-1k dataset.

- `test.py`: run this file to evaluate your results. Replace this nparray`your_result` in line 5 by your output:

  ``` python
  ...
  infer = Eval()
  your_results = ... # replace by your results
  given_times = [10,15]
  ...
  ```

### Additional Resource

Tang et al. "Multi-modal Segment Assemblage Network for Ad Video Editing with Importance-Coherence Reward"

### Acknowledgement

We thank Tencent Inc. and SUSTech for support to the project.

### License
