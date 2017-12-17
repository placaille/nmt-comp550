import os
from collections import OrderedDict

class Experiment():
    def __init__(self, path):
	self.id = path.split('/')[-1]
	self.args = OrderedDict()
	self.path = path
	self.parse_args()
	self.parse_log()

    def parse_args(self):
    	with open(os.path.join(self.path, 'args.info'), 'rb') as f: 
	    lines = f.read().splitlines()
	i = 0
	while i < len(lines) - 2:
	    line = lines[i]
	    if '\'' in line: 
		name = line.split('\'')[1]
		i += 2
		self.args[name] = lines[i]
	    i += 1

    def parse_log(self):
        with open(os.path.join(self.path, 'log.txt'), 'rb') as f:
            lines = f.read().splitlines()
        self.test_2016 = float(lines[-3].split('|')[2].split(' ')[-2])
	self.test_2017 = float(lines[-2].split('|')[2].split(' ')[-2])

    def __lt__(self, other):
       return self.test_2016 < other.test_2016

    def __str__(self):
       str = ''
       for arg, value in self.args.items():
	    str += '{},'.format(value)
       return '{}, {}, {}\r\n'.format(str, self.test_2016, self.test_2017) 

    def print_args(self):
	str = ''
	import pdb; pdb.set_trace()
        for key in self.args.keys():
	    str += '{},'.format(key)
	str += 'test_2016, test_2017\r\n'
	return str


exp_dir = './out'
exps = []

for i in range(1,600):
    print i
    try : 
        exp = Experiment(os.path.join(exp_dir, str(i)))
        exps.append(exp)
    except: 
	pass

exps.sort()
with open('exp_log.txt', 'wb') as f: 
    f.write(exps[0].print_args())
    for exp in exps:
        f.write(exp.__str__())
	    
