import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import xml.etree.ElementTree as ET
from gensim.models import FastText
from collections import defaultdict, Counter
from gensim.test.utils import datapath
# from gensim.models.fasttext import load_facebook_vectors

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

embed_dim = 50

# print("start loading...")
# en_vec = load_facebook_vectors('/Users/yvonna/Documents/cse 481 N nlp capstone/NLP-Capstone/cc.en.300.bin')
# fr_vec = load_facebook_vectors('/Users/yvonna/Documents/cse 481 N nlp capstone/NLP-Capstone/cc.fr.300.bin')
# print("finish loading model")




files = glob.glob('fr/*')
focus_word = set([filename.split("/")[-1][0:-9] for filename in files])
print(focus_word)

input_file = open('input.txt')
input_sentences = input_file.readlines()

align_file = open("forward.align")
alignments = align_file.readlines()

final_dict = defaultdict(Counter)

window = 2



def generate_embedding_from_list(model, words):
    embed = np.zeros(embed_dim)
    for word in words:
        embed += model.wv[word]
    return embed

common_en = []
common_fr = []
for i in range(len(input_sentences)):
    parts = input_sentences[i].split("|||")
    en = parts[0].split()
    fr = parts[1].split()
    common_en.append(en)
    common_fr.append(fr)

# print("start training...")
# en_model = FastText(size=embed_dim, window=3, min_count=1)  # instantiate
# en_model.build_vocab(sentences=common_en)
# en_model.train(sentences=common_en, total_examples=len(common_en), epochs=10)  # train
# en_model.save('en_model.bin')
# print("finish en training...")
# fr_model = FastText(size=embed_dim, window=3, min_count=1)  # instantiate
# fr_model.build_vocab(sentences=common_fr)
# fr_model.train(sentences=common_fr, total_examples=len(common_fr), epochs=10)  # train
# en_model.save('fr_model.bin')
# print("finish fr training...")
en_model = FastText.load('en_model.bin')
fr_model = FastText.load('fr_model.bin')
print("finish loading model...")


en_arr = []
fr_arr = []
for i in range(len(input_sentences)):
    parts = input_sentences[i].split("|||")
    en = parts[0].split()
    fr = parts[1].split()
    cur_align = alignments[i].split()
    for j, word in enumerate(en):
        if word in focus_word:
            align_pointer = 0
            phrase = []
            while align_pointer < len(cur_align):
                if int(cur_align[align_pointer].split('-')[0]) != j:
                    align_pointer += 1
                else:
                    while align_pointer < len(cur_align) and int(
                            cur_align[align_pointer].split('-')[0]) == j:
                        phrase.append(fr[int(cur_align[align_pointer].split('-')[1])])
                        align_pointer += 1
            if phrase:
                sentence_embed = generate_embedding_from_list(en_model, en[min(0, j-window):j+window])
                aligned_phrase_embed = generate_embedding_from_list(fr_model, phrase)
                en_arr.append(sentence_embed)
                fr_arr.append(aligned_phrase_embed)
en_arr = np.array(en_arr)
fr_arr = np.array(fr_arr)
print("finish reading input file")


model = LinearRegressionModel(embed_dim, embed_dim)

criterion = nn.MSELoss()# Mean Squared Loss
l_rate = 0.001
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent
epochs = 500

for epoch in range(epochs):

    epoch +=1
    #increase the number of epochs by 1 every time
    inputs = Variable(torch.from_numpy(en_arr).float())
    labels = Variable(torch.from_numpy(fr_arr).float())

    #clear grads as discussed in prev post
    optimiser.zero_grad()
    #forward to get predicted values
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()# back props
    optimiser.step()# update the parameters
    print('epoch {}, loss {}'.format(epoch, loss.item()))
torch.save(model, 'we')


data_files = glob.glob('data/*')
for file in data_files:
    tree = ET.parse(file)
    root = tree.getroot()
    word_root = root[0]
    word = word_root.attrib['item'][:-2]
    f = open('results/'+word+'.txt', 'w')
    for instance in word_root:
        id = instance.attrib['id']
        context = instance[0]
        input = list(context.itertext()) # sentences segments
        if input[1].startswith(word):
            embed = generate_embedding_from_list(en_model, input[0][-window:]+input[2][:window])
            output = model.forward(torch.Tensor(embed))
            label = fr_model.wv.similar_by_vector(output.detach().numpy(), topn=1)[0][0]
            f.write(word + '.n.fr ' + id + ' :: ' + label + ';\n')





