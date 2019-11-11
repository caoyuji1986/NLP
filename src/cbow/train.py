import cbow.data_prepare as dp
import cbow as cbow
import numpy as np

class Ancient():
    file_name = "../data/poem.txt"
    vocabulary_size = 5000
    embedding_size = 300
    window_size = 2
    num_sampled = 128
    step_num = 1200000
    batch_size = 128
    dst_path = '../out/word2vec'

    def train(self):
        sequence_word = dp.read_data(self.file_name)
        cbow_ = cbow.CBow(vocabulary_size=self.vocabulary_size,
              embedding_size=self.embedding_size,
              window_size=self.window_size,
              num_sampled=self.num_sampled,
                words=sequence_word)
        cbow_.create_graph()
        cbow_.train(steps=self.step_num, batch_size=self.batch_size, dst_path=self.dst_path)

    def __most_similiar(self, word, embedding, score, dict, reverse_dict):
        for i in range(len(word)):
            x = word[i]
            y = embedding[i]
            x_similiar = np.argsort(score[i]).tolist()
            x_similiar.reverse()
            for i in range(min(10, len(x_similiar))):
                print(reverse_dict[i], end=",")
            print("\n")

    def infer(self, word):
        sequence_word = dp.read_data(self.file_name)
        data,_,dictionary, reverse_dictionary = dp.build_dataset(sequence_word, self.vocabulary_size)
        cbow_ = cbow.CBow(vocabulary_size=self.vocabulary_size,
                          embedding_size=self.embedding_size,
                          window_size=self.window_size,
                          num_sampled=self.num_sampled,
                          words=sequence_word)
        cbow_.load(self.dst_path)
        x_similiar_score, x_embedding = cbow_.inference(word)
        self.__most_similiar(word, x_embedding, x_similiar_score, dictionary, reverse_dictionary)


class Modern():
    # 每行一句话，不包含标点
    file_name = "../data/weibo_chinese.txt"
    vocabulary_size = 15000
    embedding_size = 200
    window_size = 3
    num_sampled = 256
    step_num = 2000001
    batch_size = 128
    dst_path = '../out/word2vec_modern'

    def train(self):
        sequence_word = dp.read_data(self.file_name, "modern")
        cbow_ = cbow.CBow(vocabulary_size=self.vocabulary_size,
                      embedding_size=self.embedding_size,
                      window_size=self.window_size,
                      num_sampled=self.num_sampled,
                      words=sequence_word)
        cbow_.create_graph()
        cbow_.train(steps=self.step_num, batch_size=self.batch_size, dst_path=self.dst_path)

    def infer(self):
        sequence_word = dp.read_data(self.file_name, "modern")
        data, _, dictionary, reverse_dictionary = dp.build_dataset(sequence_word, self.vocabulary_size)
        cbow_ = cbow.CBow(vocabulary_size=self.vocabulary_size,
                          embedding_size=self.embedding_size,
                          window_size=self.window_size,
                          num_sampled=self.num_sampled,
                          words=sequence_word)
        cbow_.load(self.dst_path)
        cbow_.inference([1,2,3])

if __name__=="""__main__""":
    #Ancient().infer([1])
    Modern().train()