import time
import tensorflow as tf

from nmt.transformer.feature import file_based_input_fn_builder

input_fn = \
	file_based_input_fn_builder(input_file='./dat/nmt/wmt2014/train.tfrecord',
														is_training=False, drop_remainder=False)
d = input_fn({'train_batch_size': 4})
iter = tf.data.Iterator.from_structure(d.output_types, d.output_shapes)
xs, ys = iter.get_next()
test_init_op = iter.make_initializer(d)

with tf.Session() as sess:
	sess.run(test_init_op)
	
	while True:
		
		print("y_label------------------------------------")
		ret = sess.run(ys)
		for ele in ret:
			print(ele)
			
		print("x-----------------------------------------")
		ret = sess.run(xs[0])
		for ele in ret:
			print(ele)

		print("y-----------------------------------------")
		ret = sess.run(xs[1])
		for ele in ret:
			print(ele)
		time.sleep(1)
