import json

if __name__=="__main__":
    fout = open("../raw_data/all.orig",'a')
    start_index = 30000
    with open("TPdata.txt") as file:
        for line in file.readlines():
            fields = line.strip().split("\t")
            orig_id = fields[0]
            target = fields[1]
            text = fields[2]
            stance = fields[3]
            out_line = "{}\t{}\t{}\t{}\n".format(start_index,target,text,stance)
            start_index += 1
            fout.write(out_line)
    fout.close()
