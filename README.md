# Myown_sbert

论文地址：[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

数据集：LCQMC

句子对的形式
text_a	text_b	label
喜欢打篮球的男生喜欢什么样的女生	爱打篮球的男生喜欢什么样的女生	1
我手机丢了，我想换个手机	我想买个新手机，求推荐	1
大家觉得她好看吗	大家觉得跑男好看吗？	0


模型：
bert+2层隐藏层+全连接层

做句子的语义分类

![](https://img.shields.io/badge/Python-3.7.5-blue)
![](https://img.shields.io/badge/torch-1.8.0-green)
![](https://img.shields.io/badge/transformers-4.5.1-green)
![](https://img.shields.io/badge/tqdm-4.49.0-green)

##最新更新

```
模型model/sentence_bert.py——更新了模型输出向量的均值方法pooling
similarity_valdation.py——相似度验证
query_topn_search.py——相似度搜索
```

##模型训练
```bash
python train_sentence_bert.py
或者
bash run_train_sentence_bert.sh
```

##模型效果
```
最新数据采用业务数据，效果记录在
train_model_record.txt中
```

# 2. 参考
- [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
