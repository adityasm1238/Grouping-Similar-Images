from numpy import load
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image

#Loading the encodings created by encoding.py
d = load('data.npz')
da = d['arr_0']

#Reading all the files in directory
files = os.listdir('data')



#Calculating cosine similarity ang comparing images

similarity = 0.6  #groups images having similarity greater than .6 

data = {}
for i in tqdm (range(len(files)), desc="Comparing Images",unit="file"):
	for j in range(i+1,len(files)):
		cosineSimilarity = np.dot(da[i,:],da[j,:])/(np.linalg.norm(da[i,:])*np.linalg.norm(da[j,:]))
		if(cosineSimilarity>similarity):
			try:
				data[files[i]].append(files[j])
			except:
				data[files[i]] = [files[j]]

#Merging similar images in one group 
present = {}
newD = {}
for i in data:
	k = i
	try:
		if(present[i]['b']):
			k = present[i]['v']
	except:
		pass
	for j in data[i]:
		n = {}
		n['b']=True
		n['v']=k
		present[j]=n

for i in present:
	try:
		newD[present[i]['v']].append(i)
	except:
		newD[present[i]['v']] = [i]
l = 0
grpd = []
finalGroups={}
for i in newD:
	finalGroups['Group'+str(l)]=[i]
	grpd.append(i)
	for j in newD[i]:
		finalGroups['Group'+str(l)].append(j)
		grpd.append(j)
	l+=1
finalGroups['ungrouped']=[]
for f in files:
	if f not in grpd:
		finalGroups['ungrouped'].append(f)

#Ploting Group0 for testing
grp=finalGroups['Group1']
fig=plt.figure(figsize=(8, 8))
for i in range(len(grp)):
    img = Image.open("data/"+grp[i])
    ax = fig.add_subplot((len(grp)//5)+(len(grp)%5), 5, i+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(img)
plt.show()