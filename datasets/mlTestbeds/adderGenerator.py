#!/usr/bin/python
numOfPositions = 7
k=24


print "@relation 'adder"+str(numOfPositions)+"_"+str(k)+"'"

for i in range(0,numOfPositions):
	print "@attribute attr"+str(i)+" {0,1}"

for i in range(0,numOfPositions):
	print "@attribute label"+str(i)+" {0,1}"

print '@data'

for i in range(pow(2,numOfPositions),pow(2,numOfPositions+1)):
	number = bin(i)[3:]; # pairno to duadiko ari9mo i
	conc = i + k;
	conc = bin(conc)[-numOfPositions:];
	#print "number:"+number;
	#print "conc:"+conc;


	line="";
	for character in number:
			line+=","+character

	for character in conc:
		line+=","+character
	line = line[1:]
	print line
	
		
