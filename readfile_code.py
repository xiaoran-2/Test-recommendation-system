
# coding: utf-8

# In[209]:

import csv
import os
import os.path
from os import listdir
from os.path import isfile, join
from os import walk
import operator
import numpy as np
import math


# In[191]:

# 得到所有文件的列表
fs = []
for (dirpath, dirnames, filenames) in walk('Students'):
    for file in filenames:
        fs.append(dirpath + "/" + file)

fs.sort()


# In[192]:

file = 'Students/00000000.txt'
testid2peo = {} #纪录所有的题目id，对应的做过该题的人
allpeople = {} #所有人的数据字典，每一项是学生id和数据
peopletestset = {} #纪录每个用户做过的题目 


# 初始化字典，表示做过第i题的所有的人的id
for i in range(10000):
    tmp = []
    testid2peo[i] = tmp
    

def getAverage(rowdata):
    s = 0
    for i in range(len(rowdata)):
        if i%2 == 1:
            s += float(rowdata[i])
#             print(rowdata[i])
    return s / (len(rowdata)/2)

# 读取一个文件的函数
def readonefile(file):    
    example = {}
    testset = []
    with open(file,'r') as f:
        for line in f.readlines():
            rowdata = line.strip().split(' ')
            if(len(rowdata) == 1):
                continue
            source = getAverage(rowdata[1:]) # 得到加权均值
            
            # 添加做过的100个题目
            testset.append(int(rowdata[0]))
            
            # 对于每一个用户，添加这个题对应的id上
            testid2peo[int(rowdata[0])].append(file)
            
            example[int(rowdata[0])] = [source ,rowdata[1:]] #每一个例题对应一个[a,b]，a是这道题的得分，
                                                  #b是知识点和得分，偶数位是知识点，奇数位是得分
                    
#     people[file] = examples
    
    return file, example,testset

readonefile(file)


# In[193]:

# 得到所有人的数据
def getallpeople(files):
    allpeople = {}#所有人的数据字典，每一项是学生id和数据
    peopletestset = {}
    for file in files:
        key, peopledata, testset = readonefile(file)
        allpeople[file] = peopledata
        peopletestset[file] = testset # 纪录每个用户的做过的题目的id
        
    return allpeople, peopletestset
    

allpeople, peopletestset = getallpeople(fs)


# In[194]:

def cos_sim(a,b):
    a = np.array(a)
    b = np.array(b)
        
    #return {"文本的余弦相似度:":np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))}
    return np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

def eucl_sim(a,b):
    a = np.array(a)
    b = np.array(b)
    #print(a,b)
    #print(np.sqrt((np.sum(a-b)**2)))
    #return {"文本的欧几里德相似度:":1/(1+np.sqrt((np.sum(a-b)**2)))}
    return 1/(1+np.sqrt((np.sum(a-b)**2)))

def pers_sim(a,b):
    a = np.array(a)
    b = np.array(b)

    a = a - np.average(a)
    b = b - np.average(b)
    return np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


# In[198]:

# 计算所有用户任意两个人之间的相似度
def simlar(fileA,fileB,sim=cos_sim):
    '''
    对每一个人的数据，针对10000个题目，得到一个10000 * 100 的矩阵
    每一个题目得到一个行向量100维，已有的知识点对应其得分，没有的知识点得分设置为0
    求两个人的做过的相同的题目计算相似度，假设形同题目的个数是k，对每一个人得到一个k * 100的向量
    计算两个向量的余弦相似度
    '''
    # 得到用户的相同的做过的题目
#     print(set(peopletestset[fileA]))
#     print(set(peopletestset[fileB]))
    
    sametest = set(peopletestset[fileA]) & set(peopletestset[fileB])
    
#     print(sametest)
    # 根据相同的题目达到len(sametest) * 100 的向量
    fileAvector = []
    fileBvector = []
    
    for t in sametest:
        
        rowdataA = [0 for i in range(100)]
        rowdataB = [0 for i in range(100)]
        
        testinfoA = allpeople[fileA][t][1] # 得到数据信息
        testinfoB = allpeople[fileB][t][1] # 得到数据信息
        
#         print(testinfoA)
#         print(testinfoB)
        
        for i in range(len(testinfoA)):
            if i % 2 == 0:
#                 print(i,testinfoA[i],testinfoA[i+1])
                rowdataA[int(testinfoA[i])] = float(testinfoA[i+1])
             
        for i in range(len(testinfoB)):
            if i % 2 == 0:
                rowdataB[int(testinfoB[i])] = float(testinfoB[i+1])
        
        
        
        fileAvector.extend(rowdataA)
        fileBvector.extend(rowdataB)
    
    fileAvector = np.array(fileAvector)
    fileBvector = np.array(fileBvector)

#     print(fileAvector)
#     print(fileBvector)
    
    if len(fileAvector)==0 or len(fileBvector) == 0: return 0

    return sim(fileAvector,fileBvector)


# In[199]:

fileA = fs[1]
fileB = fs[13]
print(len(fs))

print(simlar(fileB,fileB))


# In[210]:

any2simlars = {}

def getany2simlar():
    any2simlars = {}
    for fi in fs:
        for fj in fs:
            simdata = simlar(fi,fj)
            if math.isnan(simdata):
                simdata = 0
            any2simlars[(fi,fj)] = simdata
    
    
    return any2simlars

any2simlars = getany2simlar()


# In[247]:

def recommendSource(fileP, i, k = 4):
    '''
    如果fileP没有做题目i，则根据做过i的人，与fileP的最相近的k个人计算得分
    '''
    usedpeople = testid2peo[i]
    
    simlarfileP = {}
    # 得到fileP与这些人的相似度
    for fd in usedpeople:
        simdata = any2simlars[(fileP,fd)]
        simlarfileP[fd] = simdata
    
    
    # 降序排序得到前k个相似度最高的人，
    sortedsimpeople =  sorted(simlarfileP.items(), key=operator.itemgetter(1), reverse= True)    
    
#     print(sortedsimpeople)
    
    # 根据前k个人进行推荐,相似度 * 这个人对这个题得到得分
    recomsource = 0
    for p in sortedsimpeople[:k]:
#         print(p[1],allpeople[p[0]][i])
        recomsource += p[1] * allpeople[p[0]][i][0]
    
    return recomsource / k


def culOneSource(fileP):
    '''
    根据fileP想似的k个用户，计算每个题目的得分
    如果fileP已经选做过题目i，直接计算得分
    否则，如果fileP没有做题目i，则根据做过i的人，与fileP的最相近的k个人计算得分
    '''
    #初始化10000道题的分数
    source = [None for i in range(10000)]
    
    for key in allpeople[fileP].keys():
        source[int(key)] = allpeople[fileP][key][0] #已经选做的题，进行设置计算好的分数
    
#     print(source)
    
    # 计算没有选过的题目一道的分数
    for i in range(10000):
        if source[i] == None:
            source[i] = recommendSource(fileP, i)
    return source
    
# culOneSource(fs[0])

def culAllSource(files):
    '''
    计算所有人没每道题的得分，每一个写入一个文件中，
    每一个文件包括10000个题的题目id，得分，错误率和推荐率
    这里我们认为，得分越高证明这个人对这个题已经熟练，
    我们的错误率和推荐率有线性相关的关系，因此这个给出每个题推荐率。
    这里我们简单表示为，
    推荐率 = 1 / (得分 * 10 + 1)
    推荐的时候，只需要推荐率做高的题目即可。
    '''
    dir = 'result/'
    headers = ['题目id','得分','推荐率']
    
    for k in range(len(fs)):
        file = fs[k]
        source = culOneSource(file)
        with open(dir+file[file.find('/')+1:file.find('.')]+'.csv', 'w',newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers) #
            for i in range(10000):
                f_csv.writerow([str(i),str(source[i]), 1 / (source[i]*10 + 1)])
        


# In[248]:

culAllSource(fs)


# In[ ]:



