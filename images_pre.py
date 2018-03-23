"""
move images to 17 files 
auther: xue
date: 2018.3.5
"""
import shutil,os

path = 'images_17flowers'
failnames = os.listdir(path)

#print(path)
for i in range(len(failnames)):
    if i%80==0:
        os.mkdir(path+'/'+str(int(i/80)+1))
    shutil.move(path+'/'+failnames[i],path+'/'+str(int(i/80)+1))