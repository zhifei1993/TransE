import argparse
import tensorflow as tf
from model import TransE
from load_data import Data


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default="data/FB15K", help="data direction")
    parser.add_argument('--embedding_dim', type=int, default=100, help="entity and relationship dimensions")
    parser.add_argument('--margin_value', type=float, default=1.0, help="margin value")
    parser.add_argument('--dis_func', type=str, default='L1', help="distance function：default L1 else L2")
    parser.add_argument('--batch_size', type=int, default=4096, help="a batch size triples for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--iteration_num', type=int, default=1000, help="number of iterations")
    args = parser.parse_args()
    print('----- Model args -----')
    print(args)
    kg = Data(data_dir=args.data_dir)
    model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                   batch_size=args.batch_size, learning_rate=args.learning_rate, dis_func=args.dis_func)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('----- Start training -----')
        for iteration in range(args.iteration_num):
            model.training_run(sess, iteration)
            if (iteration + 1) % 50 == 0:
                model.evaluate_run(sess)


if __name__ == '__main__':
    main()

# FB15k
# -----Raw-----
# -----head prediction-----
# MeanRank:289.1,Hit@10:0.429
# -----tail prediction-----
# MeanRank:186.8,Hit@10:0.492
# ------Raw Average------
# MeanRank: 238.0, Hits@10: 0.461

# -----Filter-----
# -----head prediction-----
# MeanRank:102.6,Hit@10:0.683
# -----tail prediction-----
# MeanRank:71.4,Hit@10:0.726
# ------Filter Average------
# MeanRank: 87.0, Hits@10: 0.704
