from collections import defaultdict, Counter
import glob
'''
test_file = '/Users/jiweihan/Desktop/class/19sp/cse481N/Task02-FULL-finalrelease/test/clls.test.gold'
f = open(test_file)
lines = f.readlines()
focus_word = set([line.split()[0][:-2] for line in lines])
f.close()
'''
files = glob.glob('/Users/jiweihan/Desktop/class/19sp/cse481N/wsd2-master/data/test/fr/*')
focus_word = set([filename.split("/")[-1][0:-9] for filename in files])
print(focus_word)
#print(len(words))

input_file = open('input.txt')
input_sentences = input_file.readlines()

align_file = open("/Users/jiweihan/Desktop/class/19sp/cse481N/fast_align-master/build/forward.align")
alignments = align_file.readlines()

final_dict = defaultdict(Counter)

for i in range(len(input_sentences)):
	parts =input_sentences[i].split("|||")
	en = parts[0].split()
	fr = parts[1].split()
	cur_align = alignments[i].split()	
	for j, word in enumerate(en):
		if word in focus_word:
			align_pointer = 0
			while align_pointer < len(cur_align):
				if int(cur_align[align_pointer].split('-')[0]) != j:
					align_pointer += 1
				else:
					break
			phase = ''
			while align_pointer < len(cur_align) and int(cur_align[align_pointer].split('-')[0]) == j:
				phase += fr[int(cur_align[align_pointer].split('-')[1])]
				align_pointer += 1
			if phase:
				final_dict[word][phase] += 1

max_result = dict()
for key in final_dict:
	max_result[key] = final_dict[key].most_common(1)[0][0]
print(max_result)


for test_file in files:
	gold_file = open(test_file)
	result  = open('./baseline_results/' + test_file.split("/")[-1][0:-9] +'.txt', "w")
	gold_lines = gold_file.readlines()
	for each_line in gold_lines:
		parts = each_line.split("::")
		word = parts[0].split()[0][:-5] 
		fr_match = ''
		#print(word)
		if word in max_result:
			#print("we have a match")
			fr_match =  max_result[word]
		result.write(parts[0].split()[0] +" "+ parts[0].split()[1] + " :: " + fr_match + ";\n")
	result.close()
	gold_file.close()



