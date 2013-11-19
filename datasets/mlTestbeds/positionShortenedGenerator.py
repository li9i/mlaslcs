#!/usr/bin/python
numOfAttributes = 7
numOfLabels = 3


print "@relation 'position"+str(numOfAttributes)+"'"

for i in range(0,numOfAttributes):
	print "@attribute attr"+str(i)+" {0,1}"

for i in range(0,numOfLabels):
	print "@attribute label"+str(i)+" {0,1}"

print '@data'

for i in range(pow(2,numOfAttributes),pow(2,numOfAttributes+1)):
	number = bin(i)[3:]
	line="";
	for character in number:
		line+=","+character
		
	position = numOfAttributes 
	for character in number:
		if character=='1':
			break
		position -= 1
	line += ',' + ','.join(bin(position)[2:].zfill(numOfLabels))
	line = line[1:]
	print line
	
		
