import glob

fout=open("out.csv","a")
# first file:
fout.write("clip,seizure,early\n")

files_list = glob.glob('*.csv')

for file_item in files_list:
    f = open(file_item)
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() 
fout.close()
