import tensorflow as tf
import numpy as np

class model():
    def __init__(self, args):

        with tf.variable_scope("MemN2N"):
            self.q = tf.placeholder(dtype=tf.int32, shape=[None, args.vocab_size],  name="q_vector")
            self.support_s = tf.placeholder(dtype=tf.int32, shape=[args.support_size, args.vocab_size], name="support_sentences")
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, args.label_num], name="label")

            q = identity(self.q, name="q")
            for l in args.layer_num:
                u_ = tf.layers.dense(q, self.args.embedding_size, name="embedding_B_{}".format(l))
                range_u_ tf.reshape(tf.tile(u_, args.support_size), (None, args.support_size, args.embedding_size))
                
                embedding_weight_A = tf.get_variable("embedding_weight_A_{}".format(l), shape=(args.vocab_size, args.embedding_size), dtype=tf.float32)
                m_ = tf.matmul(self.support_s, embedding_weight_A)

                embedding_weight_C = tf.get_variable("embedding_weight_C_{}".format(l), shape=(args.vocab_size, args.embedding_size), dtype=tf.float32)
                c_ = tf.matmul(self.support_s, embedding_weight_C)

                pi_ = tf.nn.softmax(tf.multiply(m_, range_u_))
                q = tf.multiply(c_, pi)

            logits = tf.layers.dense(q, args.label_num, name="logit")
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)

            self.output = tf.nn.softmax(logits)

    def train(self):
        pass
