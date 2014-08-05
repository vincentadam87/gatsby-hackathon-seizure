import glob

output_fname = "out.csv"
fout=open(output_fname,"a")
# first file:
fout.write("clip,seizure,early\n")

files_list = glob.glob('*.csv')
files_list2 = [i for i in files_list if not i==output_fname]

for file_item in files_list2:
    f = open(file_item)
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() 
fout.close()
