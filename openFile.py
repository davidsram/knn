import os
#文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        returnVect.extend( lineStr.strip())
    return [int(x) for x in returnVect]

# print(img2vector('C:/Users/user/trainingDigits/0_1.txt'))
