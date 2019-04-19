import os
directory = "/Users/jiweihan/Desktop/class/19sp/cse481N/europarl-extract-master/output/parallel/EN-FR/tab"
combined = open("input.txt","w+")
for filename in os.listdir(directory):
    if filename.endswith(".tab"):
        # print(os.path.join(directory, filename))
        with open(directory  +"/"+ filename) as f:
            for x in f.readlines():
                content = x.replace('\t', ' ||| ')
                combined.write(content)
        #print(content)
        f.close()
        continue
    else:
        continue
combined.close()

