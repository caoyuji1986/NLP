import copy
import json

import six
import tensorflow as tf

from ner.model.crf_layer import crf_layer, crf_decode


class RNNNerConfig:
	
	def __init__(self,
	             embedding_fine_tuning=True,
	             hidden_size=768,
	             embedding_keep_prob=0.5,
	             rnn_keep_prob=0.5,
	             rnn_cell_type='lstm',
	             loss_type='crf',
	             vocab_size=21128,
	             embedding_size=384
	             ):
		self.embedding_fine_tuning = embedding_fine_tuning
		self.hidden_size = hidden_size
		self.embedding_keep_prob = embedding_keep_prob
		self.rnn_keep_prob = rnn_keep_prob
		self.rnn_cell_type = rnn_cell_type
		self.loss_type = loss_type
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
	
	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = RNNNerConfig()
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config
	
	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
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


class RNNNer:
	
	def __init__(self,
	             embedding_fine_tuning,
	             embedding_matrix,
	             vocab_size,
	             embedding_size,
	             input_ids,
	             input_mask,
	             labels,
	             tags_num,
	             hidden_size,
	             embedding_keep_prob,
	             rnn_keep_prob,
	             rnn_cell_type,
	             loss_type,
	             mode
	             ):
		
		self._input_ids = input_ids
		self._input_mask = input_mask
		self._labels = labels
		self._tags_num = tags_num
		self._hidden_size = hidden_size
		
		self._embedding_keep_prob = embedding_keep_prob
		self._rnn_keep_prob = rnn_keep_prob
		
		self._rnn_cell_type = rnn_cell_type
		self._loss_type = loss_type
		self._mode = mode
		
		with tf.name_scope('embedding'):
			if embedding_fine_tuning:
				if embedding_matrix is not None:
					self._embedding_matrix = tf.Variable(initial_value=embedding_matrix,
					                                     dtype=tf.float32,
					                                     name="embeddings")
				else:
					self._embedding_matrix = tf.get_variable(name="embeddings",
					                                         dtype=tf.float32,
					                                         shape=[vocab_size, embedding_size],
					                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
			
			
			else:
				self._embedding_matrix = tf.constant(value=embedding_matrix,
				                                     dtype=tf.float32,
				                                     name="embeddings")
	
	def id_2_embedding(self, x):
		
		with tf.device('/cpu:0'):
			return tf.nn.embedding_lookup(params=self._embedding_matrix, ids=x)
	
	def loss_layer(self, logits, labels):
		'''
		:return: loss->[]
		:return: pred_ids->B x S
		'''
		
		if self._loss_type == 'crf':
			tf.logging.info("loss type:  crf")
			sequence_length = tf.reduce_sum(input_tensor=self._input_mask, axis=-1)
			loss, transition = crf_layer(logits=logits, labels=labels, num_labels=self._tags_num, seq_len=sequence_length)
			pred_ids, best_score = crf_decode(potentials=logits,
			                                  transition_params=transition,
			                                  sequence_length=sequence_length)
			return loss, pred_ids, best_score
		
		elif self._loss_type == 'softmax':
			tf.logging.info("loss type:  softmax")
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
			per_example_loss = per_example_loss * tf.cast(x=self._input_mask, dtype=tf.float32)
			loss = tf.reduce_sum(input_tensor=per_example_loss) / tf.cast(x=tf.reduce_sum(input_tensor=self._input_mask),
			                                                              dtype=tf.float32) * 100
			logits = tf.nn.softmax(logits=logits, axis=-1)
			pred_ids = tf.argmax(input=logits, axis=-1)
			pred_ids = tf.cast(x=pred_ids, dtype=tf.int32)
			pred_ids = pred_ids * self._input_mask;
			return loss, pred_ids, tf.reduce_max(input_tensor=logits, axis=-1)


class BiRNNNer(RNNNer):
	LAYER_NAME = 'BiRNNNer'
	
	def __init__(self,
	             embedding_fine_tuning,
	             embedding_matrix,
	             vocab_size,
	             embedding_size,
	             input_ids,
	             input_mask,
	             labels,
	             tags_num,
	             hidden_size,
	             embedding_keep_prob,
	             rnn_keep_prob,
	             rnn_cell_type,
	             loss_type,
	             mode
	             ):
		RNNNer.__init__(self,
		                embedding_fine_tuning,
		                embedding_matrix,
		                vocab_size,
		                embedding_size,
		                input_ids,
		                input_mask,
		                labels,
		                tags_num,
		                hidden_size,
		                embedding_keep_prob,
		                rnn_keep_prob,
		                rnn_cell_type,
		                loss_type,
		                mode)
	
	def __str__(self):
		return BiRNNNer.LAYER_NAME
	
	def __build_component(self):
		with tf.name_scope('build'):

			keep_prob = self._rnn_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
			cell = tf.contrib.rnn.BasicLSTMCell(self._hidden_size)
			self._lstm_fw = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
			cell = tf.contrib.rnn.BasicLSTMCell(self._hidden_size)
			self._lstm_bw = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	
	def __make_graph(self):
		'''
		:return: logits->B x S x T
		'''
		# B x S -> B x S x E
		embedding_input_ids = self.id_2_embedding(self._input_ids)
		keep_prob = self._embedding_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
		embedding_input_ids = tf.nn.dropout(embedding_input_ids, keep_prob=keep_prob)
		
		sequence_length = tf.reduce_sum(input_tensor=self._input_mask, axis=-1)
		
		with tf.variable_scope('bidirection'):
			outputs_, _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self._lstm_fw,
				cell_bw=self._lstm_bw,
				inputs=embedding_input_ids,
				dtype=tf.float32,
				sequence_length=sequence_length
			)
			# [[B x S x H]_1,[B x S x H]_2] - > B x S x 2*H
			outputs__ = tf.concat(axis=-1, values=[outputs_[0], outputs_[1]])
			output = tf.reshape(outputs__, shape=[-1, self._hidden_size * 2])
			
			# B*S x 2*H -> B*S x H
			hidden = tf.layers.dense(inputs=output, units=self._hidden_size, activation=tf.tanh,
			                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			                         bias_initializer=tf.zeros_initializer())
		with tf.name_scope('project'):
			sequence_length_ = self._input_ids.get_shape().as_list()[1]
			# B*S x 2 -> B*S x T
			logits = tf.layers.dense(inputs=hidden, units=self._tags_num,
			                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			                         bias_initializer=tf.zeros_initializer())
			# B*S x T -> B x S x T
			logits_ = tf.reshape(tensor=logits, shape=[-1, sequence_length_, self._tags_num])
		
		return logits_
	
	def create_model(self):
		'''
		:return: loss->[]
		:return: pred_ids->B x S
		'''
		
		with tf.name_scope(UniRNNNer.LAYER_NAME):
			self.__build_component()
			logits = self.__make_graph()
			loss, pred_ids, best_score = self.loss_layer(logits=logits, labels=self._labels)
			return loss, pred_ids, best_score


class UniRNNNer(RNNNer):
	LAYER_NAME = 'UniRNNNer'
	
	def __init__(self,
	             embedding_fine_tuning,
	             embedding_matrix,
	             vocab_size,
	             embedding_size,
	             input_ids,
	             input_mask,
	             labels,
	             tags_num,
	             hidden_size,
	             embedding_keep_prob,
	             rnn_keep_prob,
	             rnn_cell_type,
	             loss_type,
	             mode
	             ):
		RNNNer.__init__(self,
		                embedding_fine_tuning,
		                embedding_matrix,
		                vocab_size,
		                embedding_size,
		                input_ids,
		                input_mask,
		                labels,
		                tags_num,
		                hidden_size,
		                embedding_keep_prob,
		                rnn_keep_prob,
		                rnn_cell_type,
		                loss_type,
		                mode)
	
	def __str__(self):
		
		return UniRNNNer.LAYER_NAME
	
	def __build_base_component(self):
		
		with tf.name_scope("build_component"):
			
			if self._rnn_cell_type == 'lstm':
				self._rnn_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._hidden_size)
			elif self._rnn_cell_type == 'gru':
				self._rnn_fw = tf.nn.rnn_cell.GRUCell(num_units=self._hidden_size)
			else:
				self._rnn_fw = tf.nn.rnn_cell.BasicRNNCell(num_units=self._hidden_size)
			
	
	def __make_graph(self):
		'''
		:return: logits->B x S x T
		'''
		
		embedding_input_ids = self.id_2_embedding(self._input_ids)
		keep_prob = self._embedding_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
		embedding_input_ids = tf.nn.dropout(embedding_input_ids, keep_prob=keep_prob)
		
		sequence_length = tf.reduce_sum(input_tensor=self._input_mask, axis=-1)
		
		with tf.variable_scope('forword'):
			outputs, _ = tf.nn.dynamic_rnn(
				self._rnn_fw,
				embedding_input_ids,
				dtype=tf.float32,
				sequence_length=sequence_length
			)
			outputs = tf.reshape(outputs, [-1, self._hidden_size])
			keep_prob = self._rnn_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
			outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
		
		with tf.name_scope('project'):
			sequence_length_ = self._input_ids.get_shape().as_list()[1]
			logits = tf.matmul(a=outputs, b=self._W) + self._b
			logits = tf.reshape(logits, [-1, sequence_length_, self._tags_num])
		
		return logits
	
	def create_model(self):
		'''
		:return: loss->[]
		:return: pred_ids->B x S
		'''
		
		with tf.name_scope(UniRNNNer.LAYER_NAME):
			self.__build_base_component()
			logits = self.__make_graph()
			loss, pred_ids = self.loss_layer(logits=logits, labels=self._labels)
			return loss, pred_ids
