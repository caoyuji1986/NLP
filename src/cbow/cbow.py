import tensorflow as tf
import cbow.data_prepare as dp
import numpy as np
import random
import shutil
import os

class CBow:
    def __init__(self, vocabulary_size,
                 embedding_size,
                 window_size,
                 num_sampled,
                 words):
        #base information
        self.__vocabulary_size = vocabulary_size
        self.__embedding_size = embedding_size
        self.__window_size = window_size
        self.__num_sampled = num_sampled

        #sample
        self.__featurePlaceHolder = None
        self.__labelPlaceHolder = None
        self.__words = words

        #graph
        self.__graph = tf.Graph()
        self.__loss = None
        self.__optimizer = None
        self.__normalized_embedding = None
        self.__data_index = 0

    def _generate_batch(self, data, batch_size):
        batch = np.ndarray(shape=(batch_size, 2*self.__window_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            # skip head and tail
            while self.__data_index - self.__window_size < 0:
                self.__data_index += 1
            if (self.__data_index + self.__window_size) >= len(data):
                # start all over again
                self.__data_index = self.__window_size
            for j in range(self.__window_size):
                batch[i, j] = data[self.__data_index - self.__window_size + j]
                batch[i, j + self.__window_size] = data[self.__data_index + (j + 1)]
            labels[i, 0] = data[self.__data_index]
            self.__data_index += 1
        return batch, labels

    def create_graph(self):
        with self.__graph.as_default():
            self.__featurePlaceHolder = tf.placeholder(dtype=tf.int32, shape=[None, self.__window_size * 2])
            self.__labelPlaceHolder = tf.placeholder(dtype=tf.int32, shape=[None, 1])

            onehot_lookup_tables = tf.Variable(
                initial_value=tf.truncated_normal(shape=[self.__vocabulary_size, self.__embedding_size])
            )

            embedding = tf.nn.embedding_lookup(params=onehot_lookup_tables, ids = self.__featurePlaceHolder)

            projection_out = tf.reduce_mean(embedding, axis=1)

            softmax_weight = tf.Variable(initial_value=tf.truncated_normal(
                shape=[self.__vocabulary_size, self.__embedding_size]
            ))
            softmax_biases = tf.Variable(initial_value=tf.zeros([self.__vocabulary_size]))

            sampled_loss_per_batch = tf.nn.sampled_softmax_loss(
                weights=softmax_weight,
                biases=softmax_biases,
                inputs=projection_out,
                labels=self.__labelPlaceHolder,
                num_sampled=self.__num_sampled,
                num_classes=self.__vocabulary_size
            )

            self.__loss = tf.reduce_mean(sampled_loss_per_batch)
            self.__optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.__loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(onehot_lookup_tables), 1, keep_dims=True))
            self.__normalized_embedding = onehot_lookup_tables / norm

    def save(self, sess, dst_path):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        with self.__graph.as_default():
            x = tf.placeholder(dtype=tf.int32, shape=[None])
            y = tf.nn.embedding_lookup(self.__normalized_embedding, x)

            x_similiar_score = tf.tensordot(self.__normalized_embedding,tf.transpose(y), 1)

            x_tensor_info = tf.saved_model.utils.build_tensor_info(x)
            y_tensor_info = tf.saved_model.utils.build_tensor_info(y)
            x_similiar_score_info = tf.saved_model.utils.build_tensor_info(x_similiar_score)
            signature_map = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    "word": x_tensor_info
                },
                outputs={
                    "embedding": y_tensor_info,
                    "x_similiar_score": x_similiar_score_info
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            builder = tf.saved_model.builder.SavedModelBuilder(dst_path)

            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={"word_embedding": signature_map}
                                                 )
            builder.save()

    def load(self, model_path):
        self.__graph = tf.Graph()
        self.__session = tf.Session(graph=self.__graph)
        meta_graph_def = tf.saved_model.loader.load(self.__session, [tf.saved_model.tag_constants.SERVING], model_path)
        graph = self.__session.graph
        signature = meta_graph_def.signature_def
        x_tensor_name = signature['word_embedding'].inputs['word'].name
        y_tensor_predict = signature['word_embedding'].outputs['embedding'].name
        x_most_similiar = signature['word_embedding'].outputs['x_similiar_score'].name

        # 获取tensor 并inference
        self.__x_tensor = graph.get_tensor_by_name(x_tensor_name)
        self.__embedding_tensor = graph.get_tensor_by_name(y_tensor_predict)
        self.__x_most_similiar_tensor = graph.get_tensor_by_name(x_most_similiar)

    def inference(self, word):
        y, x_most_similiar = self.__session.run([self.__embedding_tensor, self.__x_most_similiar_tensor], feed_dict={
            self.__x_tensor: word
        })
        print('----------------')
        print("x: {}".format(word))
        print("y: {}".format(y))
        print("x_most_similiar_score: {} ".format(x_most_similiar.T))
        return x_most_similiar.T, y

    def show_result(self, embeddings, reverse_dictionary, top_k):
        valid_size = 16
        valid_window = 100
        valid_examples = np.array(random.sample(range(valid_window), valid_size))
        for i in range(valid_size):
            valid_example = valid_examples[i]
            valid_word = reverse_dictionary[valid_example]
            # number of nearest neighbors
            sim = embeddings.dot(embeddings[valid_example, :].T)
            nearest = (-sim).argsort()[1:top_k + 1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
            print(log)


    def train(self, steps, batch_size, dst_path):
        data,_,dictionary, reverse_dictionary = dp.build_dataset(self.__words, self.__vocabulary_size)
        with tf.Session(graph=self.__graph) as sess:
            sess.run(tf.global_variables_initializer())
            average_loss = 0.0
            for step in range(steps):
                batch_data, batch_label = self._generate_batch(data, batch_size)
                feed_dict = {
                    self.__featurePlaceHolder: batch_data,
                    self.__labelPlaceHolder: batch_label
                }
                _, loss = sess.run([self.__optimizer, self.__loss], feed_dict=feed_dict)
                average_loss += loss
                if step % 3000 == 0:
                    if step > 0:
                        average_loss = average_loss / 3000
                    print("Average loss at step %d: %f" % (step, average_loss))
                    embeddings = self.__normalized_embedding.eval()
                    self.show_result(embeddings, reverse_dictionary, 8)
            self.save(sess=sess, dst_path=dst_path)
