# 1. 介绍
对于word2vec模型，能够用作词向量，为下游任务服务。

训练命令：
`
python train.py -c "/content/drive/My Drive/cbxx/data/corpus/corpus.txt" -d 512 -l en -u "data/en/analogy-19568.txt" -v "data/en/wordsim-353.txt:data/en/simlex-999.csv:data/en/wordsim_relatedness_252.txt:data/en/wordsim_similarity_203.txt:data/en/MEN_sim-3000.txt:data/en/SCWS-2003.csv" -m 10
`

测试命令：
`
python test.py -u "data/en/analogy-19568.txt" -v "data/en/wordsim-353.txt:data/en/simlex-999.csv:data/en/wordsim_relatedness_252.txt:data/en/wordsim_similarity_203.txt:data/en/MEN_sim-3000.txt:data/en/SCWS-2003.csv"
`

# 2. 命令行参数说明
命令行参数
-----------------
参数|  说明  |  默认值
----|-------|-------
-m  | 最小词频阈值| 10|
-w  | 工人数|  8|
-y  | 窗口大小  | 5|
-d  |词向量维度 | 1024|
-l  |是否分词   | en(不分词）
-i  |Hierachical Softmax| 1(是)|
-n  |负采样数量  |  5|
-e  |训练轮数 | 1|
-s  | 是否Skipgram（0表示CBOW）| 1（是）|
-z  | 最大词表数量 | 40000|

测试命令行参数
-----------------
参数|  说明  |  默认值
----|-------|-------
-u |类比测试文件| <google analogy>
-v |词相似度测试文件（支持英文冒号分割多个文件）| <wordsim353>

# 3. 训练方法
采用了实践项目提供的数据语料，语言为英文，大小为2GB。

# 4. 超参数
使用了Word2Vec提到的Skip-Gram模型，训练1轮，使用Hierarchical Softmax的训练方式，
维度d为512，词表最小词频数为10，最大词表大小为40000，其他值采用默认值。

# 5. 模型效果
最终词表数为15955。
在wordsim353测试集上，准确率达到了62.48%。
在Google Analogy测试集上，准确率达到了61.72%。

#6. 文件说明

evaluate.py 验证工具包
train.py  模型训练文件  
test.py   模型测试文件
env.tar.gz  python环境依赖包
word2vec.txt 词向量文件
README.md    实验报告


#7. 额外数据

额外测评文件在data目录下。

Word2Vec模型测评
-----------------------------------
|测评类型 | 测评名称  |   测评集数量  |
--------|----------|-------------
|analogy     |  analogy-19544(5-8869 semantic\9-10675 syntactic) | 19544 |
|analogy     |  bats-49000（4-10-1225） | 49000|
|analogy     |  semeval2012-T2-64442(79) |  64442|
|analogy     |  MEN-13500 | 13500|
| sum-analogy |   |   146486|
|similarity  |  SimLex-999  |   999 |
|similarity  |  MEN-3000   |  3000|
|similarity    | SCWS-2003 | 2003|
|similarity    | wordsim-353 |  353|
|similarity   | wordsim-related-252 |  252|
|similarity   | wordsim-similarity-203 | 203|
| sum-similarity   |    |   6801|


# 参考文献
[1] Mikolov, Tomas, et al. "Distributed Representations of Words and Phrases and their
Compositionality." NeurIPS 2013.
[2] Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in Vector Space." ICLR
2013.