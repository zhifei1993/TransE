import os
import random
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Data:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        entity_filename = "entity2id.txt"
        relation_filename = "relation2id.txt"
        train_triples_filename = "train2id.txt"
        valid_triples_filename = "valid2id.txt"
        test_triples_filename = "test2id.txt"

        entity_data = self.pd_load_data(entity_filename)
        self.entity_dict = dict(zip(entity_data[0], entity_data[1]))
        self.entity_num = len(self.entity_dict)
        self.entity_id = list(self.entity_dict.values())

        relation_data = self.pd_load_data(relation_filename)
        self.relation_dict = dict(zip(relation_data[0], relation_data[1]))
        self.relation_num = len(self.relation_dict)

        train_data = self.pd_load_data(train_triples_filename, sep=" ")
        self.train_triples = list(zip(train_data[0], train_data[1], train_data[2]))
        self.train_triples_num = len(self.train_triples)
        self.train_pool = set(self.train_triples)

        valid_data = self.pd_load_data(valid_triples_filename, sep=" ")
        self.valid_triples = list(zip(valid_data[0], valid_data[1], valid_data[2]))
        self.valid_triples_num = len(self.valid_triples)

        test_data = self.pd_load_data(test_triples_filename, sep=" ")
        self.test_triples = list(zip(test_data[0], test_data[1], test_data[2]))
        self.test_triples_num = len(self.test_triples)

        self.all_triples = set(self.train_triples + self.valid_triples + self.test_triples)

        print('----- Data size -----')
        print('实体数: {}'.format(self.entity_num), '关系数: {}'.format(self.relation_num))
        print('训练集: {}'.format(self.train_triples_num),
              '验证集: {}'.format(self.valid_triples_num),
              '测试集: {}'.format(self.test_triples_num))

    def pd_load_data(self, data_filename, sep='\t'):
        pd_data = pd.read_table(os.path.join(self.data_dir, data_filename), header=None, sep=sep)
        return pd_data

    def next_batch(self, batch_size):
        random_triples_id = np.random.permutation(self.train_triples_num)
        start = 0
        while start < self.train_triples_num:
            end = min(start + batch_size, self.train_triples_num)
            yield [self.train_triples[i] for i in random_triples_id[start:end]]
            start = end

    def generate_pos_neg_triples(self, pos_triple):
        neg_triple = []
        for (head, tail, relation) in pos_triple:
            prob = np.random.random()
            while True:
                if prob <= 0.5:
                    head = random.choice(self.entity_id)
                else:
                    tail = random.choice(self.entity_id)
                # if (head, tail, relation) not in self.train_triples：
                if (head, tail, relation) not in self.train_pool:
                    break
            neg_triple.append((head, tail, relation))
        return pos_triple, neg_triple
