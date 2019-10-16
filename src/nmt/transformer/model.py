# coding:utf8

import copy
import json
import math
import six
import numpy as np
import tensorflow as tf

from common.base import get_shape_list, layer_normalize


class TransformerConfig(object):
	"""Configuration for `TransformerModel`."""
	
	def __init__(self,
		vocab_size=37000,
		hidden_size=512,
		num_hidden_layers=6,
		attention_size=64,
		position_wise_feed_forward_size=2048,
		embedding_dropout_prob=0.1,
		sub_layer_dropout_prob=0.1,
		max_position_embeddings=512,
		init_std=0.02):
		"""
			vocab_size: bpe词表的大小, nmt 一定要用bpe
			hidden_size: 隐层的宽度 d_model
			num_hidden_layers: 编解码的层数 h
			attention_size: multi-head attention attention size d_k or d_v
			position_wise_feed_forward_size: feadforward 层的宽度 dff
			embedding_dropout_prob: Embedding dropout
			sub_layer_dropout_prob: Sublayer dropout
			max_position_embeddings: 最大的position 的长度
			init_std: 变量初始化参数
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.attention_size = attention_size
		self.position_wise_feed_forward_size = position_wise_feed_forward_size
		self.embedding_dropout_prob = embedding_dropout_prob
		self.sub_layer_dropout_prob = sub_layer_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.init_std = init_std
		
	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `TransformerConfig` from a Python dictionary of parameters."""
		config = TransformerConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config
	
	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `TransformerConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))
	
	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output
	
	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Transformer:
	"""
	论文 Attention is all your need 的一个实现
	"""
	
	def get_pos_embedding_lookup_tbl(self):
		
		return self._pos_embedding_lookup_tbl
	
	def __init__(self, config, mode=tf.estimator.ModeKeys.TRAIN):
		
		self._memory = None
		self._y_softmax = None
		self._logits = None
		self._loss = None
		
		self._config = config
		self._mode = mode
		self._debug_var = None
		
		with tf.variable_scope(name_or_scope='word_embedding', reuse=tf.AUTO_REUSE):
			"""
			[NOTICE1] embedding table 必须使用xavier_initializer
			[NOTICE2] token0 必须初始化成0向量
			"""
			embedding_lookup_tbl = tf.get_variable(name='embedding_lookup_tbl',
																								 shape=[self._config.vocab_size, self._config.hidden_size],
																								 dtype=tf.float32,
																								 initializer=tf.contrib.layers.xavier_initializer())
																								 #initializer=tf.truncated_normal_initializer(stddev=self._config.init_std))
			zero_emb = tf.zeros(shape=[1, self._config.hidden_size], dtype=tf.float32)
			self._embedding_lookup_tbl = tf.concat(values=[zero_emb, embedding_lookup_tbl[1:]], axis=0)

		with tf.variable_scope(name_or_scope='position_embedding_tbl', reuse=tf.AUTO_REUSE):
			self._pos_embedding_lookup_tbl_encoder = self.__position_embedding(self._config.max_position_embeddings,
																															 self._config.hidden_size, "encoder")
			self._pos_embedding_lookup_tbl_decoder = self.__position_embedding(self._config.max_position_embeddings,
																															 self._config.hidden_size, "decoder")

	def get_debug_var(self):

		return self._debug_var

	@staticmethod
	def __make_mask_by_value(x):
		'''
		:param x: tensor with dtype is tf.int32
		:return:	[1,1, ..., 1, 0,0, ... , 0]
		'''
		zeros = tf.zeros_like(tensor=x, dtype=tf.int32)
		ones = tf.ones_like(tensor=x, dtype=tf.int32)
		x_mask = tf.where(condition=tf.equal(x=x, y=zeros), x=zeros, y=ones)
		return x_mask
	
	@staticmethod
	def __position_embedding(maxlen, embeding_size, name="encoder"):
		"""
		:param maxlen: max_len for position embedding
		:param embeding_size: position embedding size
		:return: position embedding table [max_len X embedding_size]
		"""
		pos_emb = np.zeros(shape=[maxlen, embeding_size], dtype=np.float)
		wav_func = [math.sin, math.cos]
		d_model = float(embeding_size)
		for pos_ in range(maxlen):
			pos = float(pos_)
			for i_ in range(embeding_size):
				i = float(i_)
				x = pos / 10000.0**((i - i_ % 2) / d_model)
				pos_emb[pos_][i_] = wav_func[i_ % 2](x)
		
		pos_embedding_lookup_tbl = tf.Variable(initial_value=pos_emb, dtype=tf.float32, name=name)
		return pos_embedding_lookup_tbl
	
	@staticmethod
	def __label_smoothing(inputs, epsilon=0.1):
		'''
		使用均匀分布作为label smoothing
		inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
		epsilon: Smoothing rate.
		'''
		inputs = tf.cast(x=inputs, dtype=tf.float32)
		num_chanels = inputs.get_shape().as_list()[-1]	# number of channels
		return ((1.0 - epsilon) * inputs) + (epsilon / num_chanels)
	
	def __add_and_norm(self, x, y, scope):
		
		# HERE DIFFERENT FROM PAPER
		#keep_prob = 1.0 - self._config.sub_layer_dropout_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
		#y = tf.nn.dropout(x=y, keep_prob=keep_prob) # sub-layer dropout
		ret = y + x
		ret = layer_normalize(inputs=ret, scope=scope)
		return ret

	@staticmethod
	def scaled_dot_product_attention(k, q, v, mask_k, mask_q, mask_v, attention_size, attention_future):

		return Transformer.__scaled_dot_product_attention(k, q, v, mask_k, mask_q, mask_v, attention_size, attention_future)

	@staticmethod
	def __scaled_dot_product_attention(k, q, v, mask_k, mask_q, mask_v,
	                                   attention_size, attention_dropout, training=True, attention_future=True):
		# Input:
		# 	k,q,v: B x S x Ha
		# 	mask_k,mask_q,mask_v: B x S
		# Return:
		#	B x S x Ha
		
		def attention_mask_before_softmax(matrix, from_mask, to_mask, attention_future=True):
			"""make sure query does not attention to key positions with value of <P>"""
			to_mask = tf.cast(x=to_mask, dtype=tf.float32)
			attention_adder = (1.0 - tf.expand_dims(input=to_mask, axis=1)) * (-2.0**31+1.0)

			if attention_future == False:
				mask_matrix = tf.ones_like(matrix[0], dtype=tf.float32)
				mask_matrix = 1.0 - tf.linalg.band_part(input=mask_matrix, num_lower=-1, num_upper=0)
				mask_matrix = tf.expand_dims(input=mask_matrix, axis=0) * (-2.0**31+1.0)
				matrix = matrix + mask_matrix # here the first dimision will be broadcast

			return matrix + attention_adder	# here attention_adder will be broadcast according to axis 1
			
		def attention_mask_after_softmax(matrix, from_mask, to_mask):
			"""make sure query positions with value of <P> do not attention to any positions of key"""
			from_mask = tf.cast(x=from_mask, dtype=tf.float32)
			attention_multiplier = tf.expand_dims(input=from_mask, axis=-1)
			return matrix * attention_multiplier # here attention_multiplier will be broadcast according to axis 2

		# QK^T
		dot_product = tf.matmul(a=q, b=k, transpose_b=True)
		# scale
		dk = tf.cast(x=attention_size, dtype=tf.float32)
		scale_dot_product = dot_product / tf.sqrt(dk)
		# mask & softmax
		scale_dot_product = attention_mask_before_softmax(matrix=scale_dot_product,
																						from_mask=mask_q, to_mask=mask_k, attention_future=attention_future)
		attention_weight_a = tf.nn.softmax(logits=scale_dot_product, axis=-1)
		#attention_weight_a = attention_mask_after_softmax(matrix=attention_weight, from_mask=mask_q, to_mask=mask_k)
		
		# HERE DIFFERENT FROM PAPER
		# attention weight dropout
		attention_weight_a = tf.layers.dropout(inputs=attention_weight_a, rate=attention_dropout, training=training)
		# attention
		attention_score = tf.matmul(a=attention_weight_a, b=v)
		return 	attention_score, dot_product, scale_dot_product, attention_weight_a, attention_weight_a
	
	def __create_encoder(self, x_input):
		
		# embedding
		x_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl, ids=x_input)
		x_id_emb *= get_shape_list(x_id_emb)[-1] ** 0.5 # IMPORTANT !!!!!
		
		# position embedding
		seq_len = get_shape_list(self._x)[1]
		x_position_emb = tf.slice(input_=self._pos_embedding_lookup_tbl_encoder, begin=[0, 0], size=[seq_len, -1])
		x_position_emb = x_position_emb * tf.cast(x=tf.expand_dims(input=self._x_mask, axis=-1), dtype=tf.float32)
		x = x_id_emb + x_position_emb	# B x S x H 请注意在生成特征的时候不要超过position embedding的最大长度
		
		# model regularization
		x = tf.layers.dropout(inputs=x, rate=self._config.embedding_dropout_prob,
		                      training=self._mode == tf.estimator.ModeKeys.TRAIN)
		
		for i in range(self._config.num_hidden_layers):
			with tf.variable_scope("encoder_layer" + str(i), reuse=tf.AUTO_REUSE):
				# multi-head attention
				attention_heads = []
				for j in range(int(self._config.hidden_size / self._config.attention_size)):
					with tf.variable_scope("attention_head" + str(j), reuse=tf.AUTO_REUSE):
						# B x S x Ha
						# dense layer use xavier_initializer  which make the input distribution the same as output
						k = tf.layers.dense(inputs=x, units=self._config.attention_size, use_bias=True, name='k')
						q = tf.layers.dense(inputs=x, units=self._config.attention_size, use_bias=True, name='q')
						v = tf.layers.dense(inputs=x, units=self._config.attention_size, use_bias=True, name='v')
						# B x S x Ha
						head_j,_,_,_,_ = self.__scaled_dot_product_attention(k=k, q=q, v=v,
						                      mask_k=self._x_mask, mask_q=self._x_mask, mask_v=self._x_mask,
																	 training=tf.estimator.ModeKeys.TRAIN==self._mode,
																	 attention_dropout=self._config.sub_layer_dropout_prob,
																	 attention_size=self._config.attention_size)
						
					attention_heads.append(head_j)
				
				# concat & project B x S X H -> B x S x H
				x_attention = tf.concat(values=attention_heads, axis=-1)
				# HERE DIFFERENT FROM PAPER
				#Wo = tf.get_variable(shape=[self._config.hidden_size, self._config.hidden_size], dtype=tf.float32, name='Wo')
				#x_attention = tf.einsum("bsh,hk->bsk", x_attention, Wo)

				# add & norm
				x_attention = self.__add_and_norm(x=x, y=x_attention, scope='an1')
				
				# position wise feed forward
				ff_out = tf.layers.dense(inputs=x_attention, units=self._config.position_wise_feed_forward_size,
				                         activation=tf.nn.relu, use_bias=True)
				ff_out = tf.layers.dense(inputs=ff_out, units=self._config.hidden_size, use_bias=True)
				
				# add & norm
				x = self.__add_and_norm(x=x_attention, y=ff_out, scope='an2')
	
		return x
	
	def __create_decoder(self, memory, y_input):
		
		# embedding
		y_id_emb = tf.nn.embedding_lookup(params=self._embedding_lookup_tbl, ids=y_input)
		y_id_emb *= get_shape_list(y_id_emb)[-1] ** 0.5
		# position embedding
		seq_len = get_shape_list(y_id_emb)[1]
		y_position_emb = tf.slice(input_=self._pos_embedding_lookup_tbl_decoder, begin=[0, 0], size=[seq_len, -1])
		y_position_emb = y_position_emb * tf.cast(x=tf.expand_dims(input=self._y_mask, axis=-1), dtype=tf.float32)
		y = y_id_emb + y_position_emb	# B x S x H

		# model regularization
		y = tf.layers.dropout(inputs=y, rate=self._config.embedding_dropout_prob,
		                      training=self._mode == tf.estimator.ModeKeys.TRAIN)
		
		for i in range(self._config.num_hidden_layers):
			with tf.variable_scope("decoder_layer" + str(i)):
				# multi-head attention for y
				attention_heads = []
				for j in range(int(self._config.hidden_size / self._config.attention_size)):
					with tf.variable_scope("attention_head_1_" + str(j)):
						q = tf.layers.dense(inputs=y, units=self._config.attention_size, use_bias=True, name='q1')
						k = tf.layers.dense(inputs=y, units=self._config.attention_size, use_bias=True, name='k1')
						v = tf.layers.dense(inputs=y, units=self._config.attention_size, use_bias=True, name='v1')
						head_j,_,_,_,_ = self.__scaled_dot_product_attention(k=k, q=q, v=v,
								mask_k=self._y_mask, mask_q=self._y_mask, mask_v=self._y_mask,
								training=tf.estimator.ModeKeys.TRAIN == self._mode,
								attention_dropout=self._config.sub_layer_dropout_prob,
			                                        attention_size=self._config.attention_size, attention_future=False)
						attention_heads.append(head_j)
					
				# concat & project B x S X H -> B x S x H
				y_attention = tf.concat(values=attention_heads, axis=-1)
				# HERE DIFFERENT FROM PAPER
				#Wo = tf.get_variable(shape=[self._config.hidden_size, self._config.hidden_size], dtype=tf.float32, name='Wo1')
				#y_attention = tf.einsum("bsh,hk->bsk", y_attention, Wo)
				
				# add & norm
				y_attention_ = self.__add_and_norm(x=y, y=y_attention, scope='an1')
				
				y_attention = y_attention_
				
				# multi-head attention with memory
				attention_heads = []
				for j in range(int(self._config.hidden_size / self._config.attention_size)):
					with tf.variable_scope("attention_head_2_" + str(j)):
						q = tf.layers.dense(inputs=y_attention, units=self._config.attention_size, use_bias=True, name='q1')
						k = tf.layers.dense(inputs=memory, units=self._config.attention_size, use_bias=True, name='k1')
						v = tf.layers.dense(inputs=memory, units=self._config.attention_size, use_bias=True, name='v1')
						head_j,_,_,_,_ = self.__scaled_dot_product_attention(k=k, q=q, v=v,
								mask_k=self._x_mask, mask_q=self._y_mask, mask_v=self._x_mask,
								training=tf.estimator.ModeKeys.TRAIN == self._mode,
						                attention_dropout=self._config.sub_layer_dropout_prob,
						                attention_size=self._config.attention_size,attention_future=True)
						attention_heads.append(head_j)
						
				y_attention = tf.concat(values=attention_heads, axis=-1)
				#Wo = tf.get_variable(shape=[self._config.hidden_size, self._config.hidden_size], dtype=tf.float32, name='Wo2')
				#y_attention = tf.einsum("bsh,hk->bsk", y_attention, Wo)
				
				# add & norm
				y_attention = self.__add_and_norm(x=y_attention_, y=y_attention, scope='an2')
				
				# feed forward
				ff_out = tf.layers.dense(inputs=y_attention, units=self._config.position_wise_feed_forward_size,
				                         activation=tf.nn.relu, use_bias=True)
				ff_out = tf.layers.dense(inputs=ff_out, units=self._config.hidden_size, use_bias=True)
				
				# add & norm
				y = self.__add_and_norm(x=y_attention, y=ff_out, scope='an3')
		
		# linear B x S x H -> B x S x V
		logits = tf.einsum('bsh,hv->bsv', y, tf.transpose(self._embedding_lookup_tbl))  #!IMPORTANT
		# softmax
		y_softmax = tf.nn.softmax(logits=logits, axis=-1)
		
		return logits, y_softmax

	def __calculate_loss(self, logits):
		"""average loss of word prediction"""
		log_probs = tf.nn.log_softmax(logits=logits, axis=-1)
		self._log_probs = log_probs[0]
		one_hot_labels = tf.one_hot(self._y_label, depth=self._config.vocab_size, dtype=tf.float32)
		one_hot_labels = self.__label_smoothing(one_hot_labels)
		per_sample_loss = -tf.reduce_sum(input_tensor=(one_hot_labels * log_probs), axis=-1)
		per_sample_loss = per_sample_loss * tf.cast(x=self._y_label_mask, dtype=tf.float32)
		loss = tf.reduce_sum(input_tensor=per_sample_loss) / tf.reduce_sum(tf.cast(x=self._y_label_mask, dtype=tf.float32))

		return loss

	def create_model(self,	x, y, y_label):
		self._x = x	# B x S [<B>, x1, x2, ... ,xn, <E>, <P>, ..., <P>]
		self._x_mask = self.__make_mask_by_value(x)	# B x S
		self._memory = self.__create_encoder(x_input=self._x)
		
		self._y = y	# B x S [<B>, y1, y2, ... , ym, <P>, ..., <P>]
		self._y_mask = self.__make_mask_by_value(y)	# B x S
		
		self._y_label = y_label	# B x S [y1, y2, ..., ym, <E>, <P>, ..., <P>]
		self._y_label_mask = self.__make_mask_by_value(y_label)	# B x S
		self._logits, self._y_softmax = self.__create_decoder(memory=self._memory, y_input=self._y)
		
		self._loss = self.__calculate_loss(logits=self._logits)
	
	def get_loss(self):
		
		return self._loss
	

	def create_encoder(self, x_placeholder):
		
		self._x = x_placeholder	# B x S [<B>, x1, x2, ... ,xn, <E>, <P>, ..., <P>]
		self._x_mask = self.__make_mask_by_value(x_placeholder)	# B x S
		return self.__create_encoder(x_input=self._x)
		
	def create_decoder(self, memory_placeholder, y_placeholder):
		
		self._y = y_placeholder	# B x S [<B>, y1, y2, ... , ym, <P>, ..., <P>]
		self._y_mask = self.__make_mask_by_value(y_placeholder)	# B x S
		self._logits, y_softmax = self.__create_decoder(memory=memory_placeholder, y_input=self._y)
		return tf.argmax(input=y_softmax, axis=-1)
