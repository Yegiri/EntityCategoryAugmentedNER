"""Data loader"""
import os
import torch
import utils
import random
import numpy as np
from transformers import BertTokenizer
from build_word_vocab import load_vocab, load_type, load_word2type
import jieba


class DataLoader(object):
    def __init__(self, data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_class, do_lower_case=False)

        with open(data_dir + '/train/sentences.txt', 'r') as f:
            train_data = f.readlines()
        char_to_ix = {}
        char_to_ix['<PAD>'] = 0
        char_to_ix['<UNK>'] = 1
        for sentence in train_data:
            sentence = sentence.strip().split()
            for word in sentence:
                if word not in char_to_ix:
                    char_to_ix[word] = len(char_to_ix)
        self.char2idx = char_to_ix
        self.word2idx, _ = load_vocab()
        self.type2idx, _ = load_type()
        self.word2type, _ = load_word2type()

    def get_word2idx(self):
        return self.char2idx

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def cut_word(self, sent):
        ori_sent = sent.replace(' ', '')
        word_list=list(jieba.cut(ori_sent, cut_all=True))
        # print(word_list)
        word_list = list(set(word_list))
        token_list = sent.split()

        # print(ori_sent)
        # print(word_list)

        B_set = [[] for i in range(len(token_list))]
        M_set = [[] for i in range(len(token_list))]
        E_set = [[] for i in range(len(token_list))]
        S_set = [[] for i in range(len(token_list))]

        B_type = [[] for i in range(len(token_list))]
        M_type = [[] for i in range(len(token_list))]
        E_type = [[] for i in range(len(token_list))]
        S_type = [[] for i in range(len(token_list))]

        for word in word_list:
            for i, char in enumerate(token_list):
                word_id = self.word2idx[word]
                type_id = self.type2idx[self.word2type[word][0]]
                Type = self.word2type[word][0]
                if char in word:
                    char_index = word.index(char)
                    if ori_sent[i - char_index : i-char_index+len(word)] == word:
                        # 如果词是单个字，则记录在S_set中
                        if len(word) == 1:
                            S_set[i].append(word_id)
                            S_type[i].append(type_id)
                        elif char_index == 0:
                            B_set[i].append(word_id)
                            B_type[i].append(type_id)
                            # B_set[i].append(word)
                            # B_type[i].append(Type)
                        elif char_index == (len(word) - 1):
                            E_set[i].append(word_id)
                            E_type[i].append(type_id)
                        else:
                            M_set[i].append(word_id)
                            M_type[i].append(type_id)

        # for i in range(len(ori_sent)):
        #     print(ori_sent[i], B_set[i], M_set[i], E_set[i], S_set[i])
        
        B_set = [i if i != [] else [0] for i in B_set]
        M_set = [i if i != [] else [0] for i in M_set]
        E_set = [i if i != [] else [0] for i in E_set]
        S_set = [i if i != [] else [0] for i in S_set]

        B_type = [i if i != [] else [0] for i in B_type]
        M_type = [i if i != [] else [0] for i in M_type]
        E_type = [i if i != [] else [0] for i in E_type]
        S_type = [i if i != [] else [0] for i in S_type]

        for i in range(len(token_list)):
            assert len(B_set[i]) == len(B_type[i])
            assert len(M_set[i]) == len(M_type[i])
            assert len(M_set[i]) == len(M_type[i])
            assert len(M_set[i]) == len(M_type[i])
        
        return B_set, M_set, E_set, S_set, B_type, M_type, E_type, S_type

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []
        B_sets = []
        M_sets = []
        E_sets = []
        S_sets = []
        B_types = []
        M_types = []
        E_types = []
        S_types = []
                    
        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                sent = line.strip()

                B_set, M_set, E_set, S_set, B_type, M_type, E_type, S_type = self.cut_word(sent)
                B_sets.append(B_set)
                M_sets.append(M_set)
                E_sets.append(E_set)
                S_sets.append(S_set)
                B_types.append(B_type)
                M_types.append(M_type)
                E_types.append(E_type)
                S_types.append(S_type)

                tokens = sent.split(' ')
                tokens_ids = [self.char2idx[i] if i in self.char2idx else 1 for i in tokens]
                sentences.append(tokens_ids)
        if tags_file != None:
            with open(tags_file, 'r') as file:
                for line in file:
                    # replace each tag by its index
                    tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                    tags.append(tag_seq)

            # checks to ensure there is a tag for each token
            assert len(sentences) == len(tags)
            for i in range(len(sentences)):
                assert len(tags[i]) == len(sentences[i])

            d['tags'] = tags

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['size'] = len(sentences)
        d['B_set'] = B_sets
        d['M_set'] = M_sets
        d['E_set'] = E_sets
        d['S_set'] = S_sets

        d['B_type'] = B_types
        d['M_type'] = M_types
        d['E_type'] = E_types
        d['S_type'] = S_types

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        
        if data_type in ['train', 'val', 'test']:
            print('Loading ' + data_type)
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        elif data_type == 'interactive':
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            self.load_sentences_tags(sentences_file, tags_file=None, d=data)   
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled
            
        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)
        
        interMode = False if 'tags' in data else True

        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size']//self.batch_size
        else:
            BATCH_NUM = data['size']//self.batch_size + 1


        # one pass over data
        for i in range(BATCH_NUM):
            # fetch sentences and tags
            if i * self.batch_size < data['size'] < (i+1) * self.batch_size:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:]]
                B_sets = [data['B_set'][idx] for idx in order[i*self.batch_size:]]
                M_sets = [data['M_set'][idx] for idx in order[i*self.batch_size:]]
                E_sets = [data['E_set'][idx] for idx in order[i*self.batch_size:]]
                S_sets = [data['S_set'][idx] for idx in order[i*self.batch_size:]]

                B_types = [data['B_type'][idx] for idx in order[i*self.batch_size:]]
                M_types = [data['M_type'][idx] for idx in order[i*self.batch_size:]]
                E_types = [data['E_type'][idx] for idx in order[i*self.batch_size:]]
                S_types = [data['S_type'][idx] for idx in order[i*self.batch_size:]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i*self.batch_size:]]
            else:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                B_sets = [data['B_set'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                M_sets = [data['M_set'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                E_sets = [data['E_set'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                S_sets = [data['S_set'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

                B_types = [data['B_type'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                M_types = [data['M_type'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                E_types = [data['E_type'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                S_types = [data['S_type'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                if not interMode:
                    tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)
            max_len_B = max([len(w) for s in B_sets for w in s])
            max_len_M = max([len(w) for s in M_sets for w in s])
            max_len_E = max([len(w) for s in E_sets for w in s])
            max_len_S = max([len(w) for s in S_sets for w in s])
            # max_token_len = 0
            max_token_len = max_subwords_len

            # B_set (batch_num, seq_len, word_num)
            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_subwords_len))
            batch_B = 0 * np.ones((batch_len, max_subwords_len, max_len_B))
            batch_M = 0 * np.ones((batch_len, max_subwords_len, max_len_M))
            batch_E = 0 * np.ones((batch_len, max_subwords_len, max_len_E))
            batch_S = 0 * np.ones((batch_len, max_subwords_len, max_len_S))

            batch_B_type = 0 * np.ones((batch_len, max_subwords_len, max_len_B))
            batch_M_type = 0 * np.ones((batch_len, max_subwords_len, max_len_M))
            batch_E_type = 0 * np.ones((batch_len, max_subwords_len, max_len_E))
            batch_S_type = 0 * np.ones((batch_len, max_subwords_len, max_len_S))
            # batch_token_starts = []
            
            # copy the data to the numpy array
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j])
                if cur_subwords_len <= max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j]
                    for k in range(cur_subwords_len):
                        cur_b_len = len(B_sets[j][k])
                        cur_m_len = len(M_sets[j][k])
                        cur_e_len = len(E_sets[j][k])
                        cur_s_len = len(S_sets[j][k])
                        batch_B[j][k][:cur_b_len] = B_sets[j][k]
                        batch_M[j][k][:cur_m_len] = M_sets[j][k]
                        batch_E[j][k][:cur_e_len] = E_sets[j][k]
                        batch_S[j][k][:cur_s_len] = S_sets[j][k]

                        batch_B_type[j][k][:cur_b_len] = B_types[j][k]
                        batch_M_type[j][k][:cur_m_len] = M_types[j][k]
                        batch_E_type[j][k][:cur_e_len] = E_types[j][k]
                        batch_S_type[j][k][:cur_s_len] = S_types[j][k]
                else:
                    batch_data[j] = sentences[j][:max_subwords_len]
                    for k in range(max_subwords_len):
                        cur_b_len = len(B_sets[j][k])
                        cur_m_len = len(M_sets[j][k])
                        cur_e_len = len(E_sets[j][k])
                        cur_s_len = len(S_sets[j][k])
                        batch_B[j][k][:cur_b_len] = B_sets[j][k]
                        batch_M[j][k][:cur_m_len] = M_sets[j][k]
                        batch_E[j][k][:cur_e_len] = E_sets[j][k]
                        batch_S[j][k][:cur_s_len] = S_sets[j][k]

                        batch_B_type[j][k][:cur_b_len] = B_types[j][k]
                        batch_M_type[j][k][:cur_m_len] = M_types[j][k]
                        batch_E_type[j][k][:cur_e_len] = E_types[j][k]
                        batch_S_type[j][k][:cur_s_len] = S_types[j][k]
            
            if not interMode:
                batch_tags = self.tag_pad_idx * np.ones((batch_len, max_token_len))
                for j in range(batch_len):
                    cur_tags_len = len(tags[j])  
                    if cur_tags_len <= max_token_len:
                        batch_tags[j][:cur_tags_len] = tags[j]
                    else:
                        batch_tags[j] = tags[j][:max_token_len]
            
            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_B = torch.tensor(batch_B, dtype=torch.long)
            batch_M = torch.tensor(batch_M, dtype=torch.long)
            batch_E = torch.tensor(batch_E, dtype=torch.long)
            batch_S = torch.tensor(batch_S, dtype=torch.long)
            batch_B_type = torch.tensor(batch_B_type, dtype=torch.long)
            batch_M_type = torch.tensor(batch_M_type, dtype=torch.long)
            batch_E_type = torch.tensor(batch_E_type, dtype=torch.long)
            batch_S_type = torch.tensor(batch_S_type, dtype=torch.long)
            if not interMode:
                batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            # batch_data, batch_token_starts = batch_data.to(self.device), batch_token_starts.to(self.device)
            batch_data = batch_data.to(self.device)
            batch_B = batch_B.to(self.device)
            batch_M = batch_M.to(self.device)
            batch_E = batch_E.to(self.device)
            batch_S = batch_S.to(self.device)

            batch_B_type = batch_B_type.to(self.device)
            batch_M_type = batch_M_type.to(self.device)
            batch_E_type = batch_E_type.to(self.device)
            batch_S_type = batch_S_type.to(self.device)

            if not interMode:
                batch_tags = batch_tags.to(self.device)
                # yield batch_data, batch_token_starts, batch_tags
                yield batch_data, batch_tags, batch_B, batch_M, batch_E, batch_S, batch_B_type, batch_M_type, batch_E_type, batch_S_type
            else:
                # yield batch_data, batch_token_starts
                yield batch_data, batch_B, batch_M, batch_E, batch_S, batch_B_type, batch_M_type, batch_E_type, batch_S_type
