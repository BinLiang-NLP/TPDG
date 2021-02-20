if __name__=="__main__":
    fout = open("tp",'w')
    for file_name in ['aet_hum.raw','antm_ci.raw','ci_esrx.raw','cvs_aet.raw']:
        with open("all.orig",encoding='utf-8', errors='ignore') as file:
            for line in file.readlines():
                fields = line.strip().split("\t")
                orig_id = fields[0]
                target = fields[1]
                text = fields[2]
                stance = fields[3]
                out_line = "{}\t{}\t{}\t{}\n".format(orig_id, target, text, stance)
                if target=="Trade Policy":
                    fout.write(out_line)
    fout.close()