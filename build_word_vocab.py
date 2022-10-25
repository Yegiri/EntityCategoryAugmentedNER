import json
import jieba
import sys
sys.path.append('./')
# from utils.arguments_parse import args
# from data_preprocessing import tools
from tqdm import tqdm

# jieba.add_word('[CLS]')
# jieba.add_word('[SEP]')
# jieba.add_word('[unused1]')


def load_data(filename):
    D = []
    with open(filename) as f:
        lines=f.readlines()
        for l in lines:
            sent = l.strip().replace(' ', '')
            D.append(sent)
    return D


def save_vocab():

    word_list=['None']
    sentences = load_data('data/msra/train/sentences.txt')
    sentences.extend(load_data('data/msra/val/sentences.txt'))
    sentences.extend(load_data('data/msra/test/sentences.txt'))
    for sent in tqdm(sentences):
        tmp_word_list=list(jieba.cut(sent,cut_all=True))
        # print(sent)
        # print(tmp_word_list)
        for word in tmp_word_list:
            if word not in word_list:
                word_list.append(word)
                
    vocab_lenth=len(word_list)
    word2id={}
    id2word={}
    for i,word in enumerate(word_list):
        word2id[word]=i
        id2word[i]=word

    with open('./data/msra/vocab.json','w',encoding='utf8') as f:
        tmp=json.dumps(word2id,ensure_ascii=False)
        f.write(tmp)

    return word2id,id2word,vocab_lenth


def load_vocab():
    with open('./data/msra/vocab.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id, len(word2id)

def load_word2type():
    with open('./data/msra/type.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id, len(word2id)

def load_type():
    with open('./data/msra/type2id.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id, len(word2id)

def build_type_vocab():
    with open('data/msra/type.json', 'r') as f:
        data = json.loads(f.read())
    count_word = {}
    vocab = ['None']
    for k, v in data.items():
        for w in v:
            if w not in vocab:
                vocab.append(w)
            if w not in count_word:
                count_word[w] = 1
            else:
                count_word[w] += 1
    
    orderd_dict = sorted(count_word.items(), key=lambda x: x[1], reverse=True)
    print(orderd_dict[:100])

    word2id={}
    id2word={}
    for i,word in enumerate(vocab):
        word2id[word]=i
        id2word[i]=word

    with open('./data/msra/type2id.json','w',encoding='utf8') as f:
        tmp=json.dumps(word2id, ensure_ascii=False)
        f.write(tmp)

    return word2id, id2word
    

if __name__=="__main__":
    # save_vocab()
    build_type_vocab()
    # word,l=load_vocab()
