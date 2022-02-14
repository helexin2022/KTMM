str=input()
sum=0
array=[]

for i in str:
    if i not in array:
        array.append(i)
        sum+=1
print(sum)
