import tensorflow as tf

def test_slice():
	
	a = tf.constant([[1.1,1.2], [2.1,2.2]])
	b = tf.slice(input_=a, begin=[0,0], size=[-1,1])

	sess = tf.Session()
	print(sess.run(a))
	print(sess.run(b))
	
def test_init():

	tf.enable_eager_execution()
	with tf.variable_scope("shared_weight_matrix"):
		embeddings = tf.get_variable('weight_mat',
																 dtype=tf.float32,
																 shape=(10, 512),
																 initializer=tf.contrib.layers.xavier_initializer())
		embeddings2 = tf.get_variable('weight_mat1',
																 dtype=tf.float32,
																 shape=(10, 512),
																 initializer=tf.truncated_normal_initializer(0.02))
		print(embeddings[8:, :])
		print(embeddings2[0])
		
def test_band_part():

	tf.enable_eager_execution()
	a = tf.constant([[1, 1, 1, 1, 1] for i in range(5)], dtype=tf.float32)
	b = 1 - tf.linalg.band_part(input=a, num_lower=-1, num_upper=0)
	print(a)
	print(b)
	
	

if __name__=='''__main__''':
	
	test_init()