#coding:utf8

import os,sys
import tensorflow as tf
import sentencepiece as spm
import numpy as np

from common.base import get_shape_list
from nmt.transformer.feature import file_based_input_fn_builder, DataProcessor, file_based_convert_examples_to_features
from nmt.transformer.flag_center import FLAGS
from nmt.transformer.model import Transformer, TransformerConfig

batch_size = int(sys.argv[3])
max_len = int(sys.argv[4])

# 1. 测试文件转换为tfrecord 文件
data_processor = DataProcessor(FLAGS.bpe_model_file)
examples = data_processor.get_test_examples(data_dir=FLAGS.data_dir)
tf_path = os.path.join(FLAGS.data_dir, 'test.tfrecord')
file_based_convert_examples_to_features(examples=examples, output_file=tf_path)

# 2. 构建模型所需的数据流
input_fn = file_based_input_fn_builder(input_file=tf_path, is_training=False, drop_remainder=False)
d = input_fn({'train_batch_size': batch_size})
iter = tf.data.Iterator.from_structure(d.output_types, d.output_shapes)
xs, ys = iter.get_next()
test_init_op = iter.make_initializer(d)
x = xs[0]
y = xs[1]
y_label = ys

# 3. 构建模型
config = TransformerConfig.from_json_file(FLAGS.model_config)
transformer = Transformer(config=config, mode=tf.estimator.ModeKeys.PREDICT)

x_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
memory = transformer.create_encoder(x_placeholder=x_placeholder)

memory_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, config.hidden_size], name='memory')
y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y')
y_predict = transformer.create_decoder(memory_placeholder=memory_placeholder, y_placeholder=y_placeholder)

# 4. 执行infer流程
sent_piece = spm.SentencePieceProcessor()
sent_piece.Load(FLAGS.bpe_model_file)

y_predict_list = list()
y_label_list = list()

with tf.Session() as sess:
	sess.run(test_init_op)
	ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
	print(ckpt_path)
	saver = tf.train.Saver()
	saver.restore(sess, ckpt_path)
	
	ii = 0
	while True:
		
		ii += 1
		if ii == 900:
			break
		x_val, y_label_val = sess.run([x, y_label])
		
		memory_val = sess.run(memory, feed_dict={x_placeholder: x_val})
		y_val_0 = np.ones(shape=[batch_size, 1], dtype=np.int32) * DataProcessor.SPECIAL_TOKEN['<S>']  # B x 1
		for i in range(max_len):
			if i == 0: y_val = y_val_0
			y_predict_val = sess.run(y_predict, feed_dict={
				memory_placeholder: memory_val,
				y_placeholder: y_val,
				x_placeholder: x_val
			}) # B x Spre
			y_val = np.concatenate([y_val_0, y_predict_val], -1)
			if y_val[0][-1] == DataProcessor.SPECIAL_TOKEN['</S>']:
				break
		
		y_predict_list.append(y_val)
		y_label_list.append(y_label_val)
		print(ii)
	
# 5. 结果写入
def post_process(y_predict):
	"""去空处理"""
	return [[int(ele_inner)
			for ele_inner in ele if ele_inner != DataProcessor.SPECIAL_TOKEN['<P>']]for ele in y_predict]


y_predict_file = sys.argv[1]
y_label_file = sys.argv[2]

with open(y_predict_file, "w") as fp_predict, open(y_label_file, "w") as fp_label:
	for i in range(len(y_predict_list)):
		predict_itm = y_predict_list[i]
		label_itm = y_label_list[i]
		predict_itm  = post_process(predict_itm)[0]
		label_itm = post_process(label_itm)[0]
		predict_itm_ = sent_piece.DecodeIds(predict_itm)
		label_itm_ = sent_piece.DecodeIds(label_itm)
		fp_predict.write(str(predict_itm_) + '\n')
		fp_label.write(str(label_itm_) + '\n')


