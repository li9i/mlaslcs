import random

for i in range(200):
	a=random.uniform(0,4);	
	noise= random.normalvariate(5,2);
	print '{0},{1},{2}'.format(noise,a,'a');
	b=random.uniform(6,10);
	noise= random.normalvariate(5,2);
	print '{0},{1},{2}'.format(noise,b,'b');

	
