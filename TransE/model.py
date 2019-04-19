import time
import tensorflow as tf
from load_data import Data


class TransE:

    def __init__(self, kg: Data, embedding_dim, margin_value, batch_size, learning_rate, dis_func):
        self.kg = kg
        self.dis_func = dis_func
        self.batch_size = batch_size
        self.margin_value = margin_value
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        self.margin = tf.placeholder(tf.float32, shape=[None])
        self.pos_triple = tf.placeholder(tf.int32, shape=[None, 3])
        self.neg_triple = tf.placeholder(tf.int32, shape=[None, 3])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.training_graph()
        self.eval_triple = tf.placeholder(tf.int32, shape=[3])
        self.evaluate_graph()

    def training_graph(self):
        bound = 6 / (self.embedding_dim ** 0.5)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[self.kg.entity_num, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[self.kg.relation_num, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))

        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, axis=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, axis=1)

        with tf.name_scope('lookup'):
            pos_h = tf.nn.embedding_lookup(self.entity_embedding, self.pos_triple[:, 0])
            pos_t = tf.nn.embedding_lookup(self.entity_embedding, self.pos_triple[:, 1])
            pos_r = tf.nn.embedding_lookup(self.relation_embedding, self.pos_triple[:, 2])
            neg_h = tf.nn.embedding_lookup(self.entity_embedding, self.neg_triple[:, 0])
            neg_t = tf.nn.embedding_lookup(self.entity_embedding, self.neg_triple[:, 1])
            neg_r = tf.nn.embedding_lookup(self.relation_embedding, self.neg_triple[:, 2])

        with tf.name_scope('loss'):
            pos_dis = pos_h + pos_r - pos_t
            neg_dis = neg_h + neg_r - neg_t
            if self.dis_func == 'L1':
                pos_score = tf.reduce_sum(tf.abs(pos_dis), axis=1)
                neg_score = tf.reduce_sum(tf.abs(neg_dis), axis=1)
            else:
                pos_score = tf.reduce_sum(tf.square(pos_dis), axis=1)
                neg_score = tf.reduce_sum(tf.square(neg_dis), axis=1)

            self.loss = tf.reduce_sum(tf.maximum(self.margin + pos_score - neg_score, 0))
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def training_run(self, session: tf.Session, iteration):
        begin_time = time.time()
        iteration_loss = 0
        for triple in self.kg.next_batch(self.batch_size):
            pos_batch, neg_batch = self.kg.generate_pos_neg_triples(triple)
            _, batch_loss = session.run(fetches=[self.train_op, self.loss],
                                        feed_dict={self.pos_triple: pos_batch,
                                                   self.neg_triple: neg_batch,
                                                   self.margin: [self.margin_value] * len(pos_batch)})
            iteration_loss += batch_loss

        end_time = time.time()
        print('Iteration:' + str(iteration) + '  ' + 'iteration loss: {:.3f}'.format(iteration_loss) +
              '  ' + 'time cost: {:.3f}'.format(end_time - begin_time))

    def evaluate_graph(self):
        with tf.name_scope('evaluate_lookup'):
            h = tf.nn.embedding_lookup(self.entity_embedding, self.eval_triple[0])
            t = tf.nn.embedding_lookup(self.entity_embedding, self.eval_triple[1])
            r = tf.nn.embedding_lookup(self.relation_embedding, self.eval_triple[2])

        with tf.name_scope('link_predict'):
            h_predict = self.entity_embedding + r - t
            t_predict = h + r - self.entity_embedding

            if self.dis_func == 'L1':
                dis_h = tf.reduce_sum(tf.abs(h_predict), axis=1)
                dis_t = tf.reduce_sum(tf.abs(t_predict), axis=1)
            else:
                dis_h = tf.reduce_sum(tf.square(h_predict), axis=1)
                dis_t = tf.reduce_sum(tf.square(t_predict), axis=1)

            _, self.h_sort_id = tf.nn.top_k(dis_h, k=self.kg.entity_num)
            _, self.t_sort_id = tf.nn.top_k(dis_t, k=self.kg.entity_num)

    def evaluate_run(self, sess: tf.Session):
        head_mean_rank_raw = 0
        head_hit_10_raw = 0
        tail_mean_rank_raw = 0
        tail_hit_10_raw = 0
        head_mean_rank_filter = 0
        tail_mean_rank_filter = 0
        head_hit_10_filter = 0
        tail_hit_10_filter = 0

        print('----- Start evaluation -----')
        begin_time = time.time()

        for triple in self.kg.test_triples:
            h_sort_id, t_sort_id = sess.run(fetches=[self.h_sort_id, self.t_sort_id],
                                            feed_dict={self.eval_triple: triple})

            a, b, c, d, e, f, g, h = self.calculate_rank(h_sort_id, t_sort_id, triple)
            head_mean_rank_raw += a
            head_hit_10_raw += b
            tail_mean_rank_raw += c
            tail_hit_10_raw += d
            head_mean_rank_filter += e
            head_hit_10_filter += f
            tail_mean_rank_filter += g
            tail_hit_10_filter += h

        end_time = time.time()
        print('-----Raw-----')
        print('-----head prediction-----')
        print('MeanRank:{:.1f},Hit@10:{:.3f}'.format(head_mean_rank_raw / self.kg.test_triples_num,
                                                     head_hit_10_raw / self.kg.test_triples_num))
        print('-----tail prediction-----')
        print('MeanRank:{:.1f},Hit@10:{:.3f}'.format(tail_mean_rank_raw / self.kg.test_triples_num,
                                                     tail_hit_10_raw / self.kg.test_triples_num))
        print('------Raw Average------')
        print('MeanRank: {:.1f}, Hits@10: {:.3f}'.format((head_mean_rank_raw / self.kg.test_triples_num +
                                                          tail_mean_rank_raw / self.kg.test_triples_num) / 2,
                                                         (head_hit_10_raw / self.kg.test_triples_num +
                                                          tail_hit_10_raw / self.kg.test_triples_num) / 2))

        print('-----Filter-----')
        print('-----head prediction-----')
        print('MeanRank:{:.1f},Hit@10:{:.3f}'.format(head_mean_rank_filter / self.kg.test_triples_num,
                                                     head_hit_10_filter / self.kg.test_triples_num))
        print('-----tail prediction-----')
        print('MeanRank:{:.1f},Hit@10:{:.3f}'.format(tail_mean_rank_filter / self.kg.test_triples_num,
                                                     tail_hit_10_filter / self.kg.test_triples_num))
        print('------Filter Average------')
        print('MeanRank: {:.1f}, Hits@10: {:.3f}'.format((head_mean_rank_filter / self.kg.test_triples_num +
                                                          tail_mean_rank_filter / self.kg.test_triples_num) / 2,
                                                         (head_hit_10_filter / self.kg.test_triples_num +
                                                          tail_hit_10_filter / self.kg.test_triples_num) / 2))
        print('time cost: {:.3f}'.format(end_time - begin_time))

    def calculate_rank(self, h_sort_id, t_sort_id, triple):
        (head, tail, relation) = triple
        head_mean_rank_raw = 1
        head_hit_10_raw = 0
        tail_mean_rank_raw = 1
        tail_hit_10_raw = 0
        head_mean_rank_filter = 1
        head_hit_10_filter = 0
        tail_mean_rank_filter = 1
        tail_hit_10_filter = 0

        h_sort_id = h_sort_id[::-1]
        t_sort_id = t_sort_id[::-1]

        for head_predict_id in h_sort_id:
            if head_predict_id == triple[0]:
                break
            else:
                head_mean_rank_raw += 1
                if (head_predict_id, tail, relation) in self.kg.all_triples:
                    continue
                else:
                    head_mean_rank_filter += 1

        for tail_predict_id in t_sort_id:
            if tail_predict_id == tail:
                break
            else:
                tail_mean_rank_raw += 1
                if (head, tail_predict_id, relation) in self.kg.all_triples:
                    continue
                else:
                    tail_mean_rank_filter += 1

        if head_mean_rank_raw <= 10:
            head_hit_10_raw = 1
        if tail_mean_rank_raw <= 10:
            tail_hit_10_raw = 1
        if head_mean_rank_filter <= 10:
            head_hit_10_filter = 1
        if tail_mean_rank_filter <= 10:
            tail_hit_10_filter = 1

        return head_mean_rank_raw, head_hit_10_raw, tail_mean_rank_raw, tail_hit_10_raw, \
               head_mean_rank_filter, head_hit_10_filter, tail_mean_rank_filter, tail_hit_10_filter
