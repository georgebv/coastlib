# An iMacros (Mozilla Firefox) script for parsing csv data from www.ncdc.noaa.gov/ulcd/ULCD
path = 'C:\iMacros'
years = list(range(2005, 2016+1))
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
names = [''.join(['94929', str(y), m]) for y in years for m in months][:-1]


# Script preamble
preamble = """
VERSION BUILD=9030808 RECORDER=FX
TAB T=1
"""

# Script worker function
loop = """
TAG POS=1 TYPE=SELECT FORM=ACTION:QCLCD ATTR=ID:tyear CONTENT=%{content}
TAG POS=1 TYPE=INPUT:SUBMIT FORM=ACTION:QCLCD ATTR=*
TAG POS=4 TYPE=INPUT:RADIO FORM=NAME:radio ATTR=NAME:which
TAG POS=1 TYPE=INPUT:SUBMIT FORM=NAME:radio ATTR=*
SAVEAS TYPE=TXT FOLDER={path} FILE={file}
BACK
BACK
"""

# Generate script body
body = [loop.format(content=name, path=path, file=name[5:])[1:] for name in names]
body = ''.join(body)

# Finalize script
with open('.\Scripts\\Quality_Controlled_LCD_.txt', 'w') as f:
    f.write(preamble[1:] + body[:-1])