import numpy as np
import tensorflow as tf

from nmt.transformer.model import Transformer, TransformerConfig


class Seq2seqMock:
	
	def __init__(self):
		self._step = tf.constant(value=-1)
	
	def get_encoder(self, x_input=None):
		
		memory = [
			[[1.1, 1.2], [2.1, 2.2]],
			[[3.1, 3.2], [4.1, 4.2]]
		]
		return tf.constant(value=memory)
	
	def get_decoder(self, momery, y_input):
		self._step += 1
		
		def true_fn0():
			a = [
				[4],
				[4]
			]
			return tf.constant(a)
		
		def false_fn0():
			
			def true_fn1():
				a = [
					[4, 5],
					[4, 6]
				]
				return tf.constant(a)
			
			def false_fn1():
				def true_fn2():
					a = [
						[4, 5, 2],
						[4, 6, 7]
					]
					return tf.constant(a)
				
				def false_fn2():
					a = [
						[4, 5, 2, 0],
						[4, 6, 7, 2]
					]
					return tf.constant(a)
				
				return tf.cond(tf.equal(self._step, 2), true_fn=true_fn2, false_fn=false_fn2)
			
			return tf.cond(tf.equal(self._step, 1), true_fn=true_fn1, false_fn=false_fn1)
		
		return tf.cond(tf.equal(self._step,0), true_fn=true_fn0, false_fn=false_fn0)


def test_scaled_dot_product_attention():
	value = [
		[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
		[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
		[[4.0, 4.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
	]
	k = tf.constant(value=value)
	q = tf.constant(value=value)
	v = tf.constant(value=value)
	mask = [
		[1,1,1],
		[1,1,0],
		[1,0,0]
	]
	mask_k = tf.constant(value=mask)
	mask_q = tf.constant(value=mask)
	mask_v = tf.constant(value=mask)
	attention_size = 3
	attention_score, dot_product, scale_dot_product, attention_weight, attention_weight_a \
	 = Transformer.scaled_dot_product_attention(k, q, v,
																					 mask_k, mask_q, mask_v,
																					 attention_size, False)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	r_val = sess.run([attention_score, dot_product, scale_dot_product, attention_weight, attention_weight_a])
	print(r_val)

def test_encoder():
	config = TransformerConfig()
	transformer = Transformer(config=config, mode=tf.estimator.ModeKeys.TRAIN)
	x_placeholder=tf.placeholder(dtype=tf.int32, shape=[3,3])
	memory = transformer.create_encoder(x_placeholder=x_placeholder)
	tvars = tf.trainable_variables()
	for tvar in tvars:
		print(tvar)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	value = [
		[1,2,3],
		[4,5,0],
		[6,0,0]
	]
	ret = sess.run(transformer.get_debug_var(), feed_dict={
		x_placeholder:value
	})
	print(ret)

def test_seq2seq_mock():
	
	seq2seq = Seq2seqMock()
	memory = seq2seq.get_encoder()
	a = [
		[1],
		[1]
	]
	y_input = tf.constant(a)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	while True:
		y_input_val = sess.run(y_input)
		print(y_input_val)
		y_input = seq2seq.get_decoder(momery=memory, y_input=y_input)
	
def test_while_loop():

	batch_size = tf.constant(8)
	num_end_inst = tf.constant(0)
	y_input = tf.constant(0)
	condition = lambda x, y: tf.less(y, batch_size)
	loop_vars = [y_input, num_end_inst]
	
	def _recurrence(y_input_, num_end_inst_):
		num_end_inst_ += batch_size
		
		return y_input_, num_end_inst_
	
	y_input, num_end_inst = \
		tf.while_loop(cond=condition, body=_recurrence, loop_vars=loop_vars)
	
	sess = tf.Session()
	ret = sess.run(num_end_inst)
	print(ret)

def test_ls():
	inputs = tf.constant([1, 3, 2, 4, 5])
	inputs = tf.one_hot(indices=inputs, depth=6)
	epsilon = 0.1
	V = inputs.get_shape().as_list()[-1]  # number of channels
	inputs_ = ((1.0 - epsilon) * inputs) + (epsilon / V)
	sess = tf.Session()
	ret = sess.run(inputs_)
	print(ret)


def test_lr():
	import tensorflow as tf
	import numpy as np
	
	x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
	y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
	w = tf.Variable(tf.constant(0.0))
	
	global_steps_ = tf.train.get_or_create_global_step()
	
	d_model = 762.0
	warmup_steps = 4000.0
	global_step = tf.cast(x=global_steps_, dtype=tf.float32)
	
	learning_rate = 1 / tf.sqrt(x=d_model) * tf.minimum(x=1 / tf.sqrt(x=global_step),
																											y=global_step / (warmup_steps ** 1.5))
	loss = tf.pow(w * x - y, 2)
	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps_)
	
	fp = open('a.txt', 'w')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(50000):
			sess.run(train_step, feed_dict={x: np.linspace(1, 2, 10).reshape([10, 1]),
																			y: np.linspace(1, 2, 10).reshape([10, 1])})
			lr = sess.run(learning_rate)
			# print(i, lr)
			fp.write(str(lr) + '\n')
	fp.close()


def test_position_embedding():
	config = TransformerConfig(vocab_size=20,
														 hidden_size=4,
														 num_attention_heads=2,
														 intermediate_size=8,
														 max_position_embeddings=10,
														 init_std=0.02)
	start = 1
	end = 2
	pad = 0
	x_ = tf.constant(value=np.array([[4, 5, end, 0, 0], [8, 9, 3, end, 0]]), dtype=tf.float32)
	
	y_ = tf.constant(value=np.array([[start, 4, 5, 0], [start, 9, 3, 0]]), dtype=tf.float32)
	y_label_ = tf.constant(value=np.array([[4, 5, end, 0], [9, 3, end, 0]]), dtype=tf.float32)
	
	transformer = Transformer(config=config, x=x_, y=y_, y_label=y_label_)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		val = sess.run(transformer.get_pos_embedding_lookup_tbl())
		print(val)
# transformer.create_model()

def test_encoder():
	
	tf.enable_eager_execution()

	config = TransformerConfig(vocab_size=20,
														hidden_size=4,
														attention_size=2,
														position_wise_feed_forward_size=8,
														max_position_embeddings=10,
														init_std=0.02)
	start = 1
	end = 2
	pad = 0
	x_ = tf.constant(value=np.array([[4, 5, end, 0, 0], [8, 9, 3, end, 0]]), dtype=tf.int32)
	
	y_ = tf.constant(value=np.array([[start, 4, 5, 0], [start, 9, 3, 0]]), dtype=tf.int32)
	y_label_ = tf.constant(value=np.array([[4, 5, end, 0], [9, 3, end, 0]]), dtype=tf.int32)
	
	transformer = Transformer(config=config)
	memory = transformer.create_encoder(x_placeholder=x_)
	print(memory)


if __name__ == """__main__""":
	
	test_encoder()
	
	
	
