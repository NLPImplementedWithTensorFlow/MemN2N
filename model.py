import tensorflow as tf
import numpy as np
import os

class model():
    def __init__(self, args):
        self.args = args

        with tf.variable_scope("MemN2N"):
            self.q = tf.placeholder(dtype=tf.float32, shape=[None, args.vocab_size],  name="q_vector")
            self.support_s = tf.placeholder(dtype=tf.float32, shape=[args.support_size, args.vocab_size], name="support_sentences")
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, args.vocab_size], name="label")

            q = tf.identity(self.q, name="q")
            for l in range(args.layer_num):
                u_ = tf.layers.dense(q, self.args.embedding_size, name="embedding_B_{}".format(l))
                range_u_ = tf.reshape(tf.tile(u_, [1, args.support_size]), shape=[-1, args.support_size, args.embedding_size])

                embedding_weight_A = tf.get_variable("embedding_weight_A_{}".format(l), shape=(args.vocab_size, args.embedding_size), dtype=tf.float32)
                m_ = tf.matmul(self.support_s, embedding_weight_A)

                embedding_weight_C = tf.get_variable("embedding_weight_C_{}".format(l), shape=(args.vocab_size, args.embedding_size), dtype=tf.float32)
                c_ = tf.matmul(self.support_s, embedding_weight_C)

                pi_ = tf.nn.softmax(tf.multiply(m_, range_u_), dim=1)

                q = tf.reduce_sum(tf.add(tf.multiply(c_, pi_), range_u_), axis=1)

            logits = tf.layers.dense(q, args.vocab_size, name="logit")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))

            self.output = tf.nn.softmax(logits)

            self.ppp = tf.identity(pi_)

    def train(self):
        optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter("saved", sess.graph)
            saver = tf.train.Saver(tf.global_variables())

            for itr in range(100):

                if itr % 200 == 0:
                    saver.save(sess, "saved/model.ckpt")
                    print("--saved model--")

                if itr == self.args.itr:
                    break
