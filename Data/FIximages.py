from os import listdir

buildings=listdir(r"buildings")

forest=listdir(r"forest")

glacier=listdir(r"glacier")

mountain=listdir(r"mountain")

sea=listdir(r"sea")

street=listdir(r"street")


mydictionary={}

for i in range(len(buildings)):
    mydictionary[buildings[i]]=0
    
for i in range(len(forest)):
    mydictionary[forest[i]]=1
    
for i in range(len(glacier)):
    mydictionary[glacier[i]]=2
    
for i in range(len(mountain)):
    mydictionary[mountain[i]]=3
    
for i in range(len(sea)):
    mydictionary[sea[i]]=4
    
for i in range(len(street)):
    mydictionary[street[i]]=5
    
keys=list(mydictionary.keys())
keys.sort()
array1=[]
array2=[]
for i in range(len(keys)):
    array1.append(keys[i])
    array2.append(mydictionary[keys[i]])
    
file1 = open("MyFile(ImageNet).csv", "a")
for i in range(len(array1)):
    file1.write(f"img_tr_{array1[i]},{array2[i]}\n")
file1.close()