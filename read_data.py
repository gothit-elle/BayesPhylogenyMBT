from bs4 import BeautifulSoup
import sys
sys.path.insert(0, '../thesis_likelihood')

with open('../thesis_likelihood\data\small_dataset.xml', 'r') as f:
	data = f.read()
Bs_data = BeautifulSoup(data, 'xml')

seqs = Bs_data.find_all('sequence')
# taxa= seqs = Bs_data.find_all('taxa')

print(seqs[0].text, seqs[0].taxon)
for n in seqs.iter('taxon'):
	print(n.attrib)