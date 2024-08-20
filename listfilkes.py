
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



# list to store files
hashes = []

# Iterate directory
for path in os.listdir(currentdir + "/csv/"):
	# check if current path is a file
	if os.path.isfile(os.path.join(currentdir + "/csv/", path)):
		if path[-5] == 'a':
			hashes.append(path[1:-5])
print(hashes)