import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
s = []
s2 = []
img = cv2.imread('C:\\Users\\blue\\Desktop\\cut_duck.jpg')
img_cut = cv2.imread('C:\\Users\\blue\\Desktop\\cut_duck1.jpg')
rows, cols = img.shape[:2]
for j in range(rows):
    for k in range(cols):
        if np.mean(img_cut[j, k]) > 220 :
            s.append(img[j, k])
        else :
            s2.append(img[j, k])

s=np.array(s)
s2=np.array(s2)
#
# print(s.shape)
# print(s2.shape)
# w1 = np.mean(s[:], axis = 0)
# w0 = np.mean(s2[:], axis = 0)
# w1_v = np.var(s[:], axis = 0)
# w0_v = np.var(s2[:], axis = 0)
#
#
#
#
# img = cv2.imread('C:\\Users\\blue\\Desktop\\full_duck.jpg')
# rows,cols = img.shape[:2]
# for i in range(rows):
#     print(i)
#     for j in range(cols):
#         w0_s = 0
#         w1_s = 0
#         for k in range(3):
#             w0_s += abs((img[i,j,k] - w0[k]) / (w0_v[k]**0.5))
#             w1_s += abs((img[i,j,k] - w1[k]) / (w1_v[k]**0.5))
#         if(w0_s > w1_s):img[i,j] = (255,255,0)
#
#
# cv2.imwrite('C:\\Users\\blue\\Desktop\\ccc.jpg',img)






# -----------------------------------高斯--------------------------------------
# 標記答案 是鴨子 = 1 不是鴨子  = 0
g= np.ones((s.shape[0],1))
g2= np.zeros((s2.shape[0],1))
# 把1和0加進原本的s中
s=np.column_stack((s,g))
s2=np.column_stack((s2,g2))
# 換成整數
# s2=np.array(s2,dtype=np.int)
# s=np.array(s,dtype=np.int)
print(s.shape)
print(s2.shape)
# 把s和s2整合成一個陣列 (n,4)(m,4) -> (n+m,4)
sn=s.shape[0]
s2n=s2.shape[0]
s_all=np.zeros((s2n+sn,4))
s_all[:sn]=s
s_all[sn:]=s2
print(s_all.shape)
# 使用skl的GaussianNB模型   .fit 是訓練模型
clf_pf = GaussianNB()
clf_pf.fit(s_all[:,:3],s_all[:,3])

img = cv2.imread('C:\\Users\\blue\\Desktop\\full_duck.jpg')
rows,cols = img.shape[:2]
img=np.array(img)

# reshape是將圖片改成可以測試的size
img1=np.reshape(img,(-1,3))
a=clf_pf.predict(img1)

# 將輸出解變成二維排列好
a=np.reshape(a,(rows,cols))
print(np.mean(a))
for i in range(rows):
    print(i)
    for j in range(cols):
        # 如果測試答案是1 將顏色改成[255,255,0]
        if a[i,j] ==1:
            img[i,j]=[255,255,0]

#a=a*255
#a=np.reshape(a,(rows,cols,1))

#cv2.imwrite('C:\\Users\\blue\\Desktop\\a.jpg',a)
cv2.imwrite('C:\\Users\\blue\\Desktop\\ccc1.jpg',img)