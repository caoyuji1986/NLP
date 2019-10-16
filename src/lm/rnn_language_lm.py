# coding:utf-8
import json
import copy

import six
import tensorflow as tf
from common.base import get_shape_list

class RNNLMConfig:
		
		def __init__(self,
		             vocab_size=21128,
		             hidden_size=768,
		             embedding_keep_prob=0.5,
		             keep_prob=0.5,
		             	):
			self.vocab_size = vocab_size
			self.hidden_size = hidden_size
			self.embedding_keep_prob = embedding_keep_prob
			self.keep_prob = keep_prob
		
		@classmethod
		def from_dict(cls, json_object):
			"""Constructs a `BertConfig` from a Python dictionary of parameters."""
			config = RNNLMConfig()
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


class RNNLM:
	
	def __init__(self, x_id, x_mask, y_id, config, mode):
		
		self._x_id = x_id
		self._x_mask = x_mask
		self._y_id = y_id
		self._config = config
		self._mode = mode
		
		self._loss = None
		self._softmax = None
	
	def create_model(self):
		input_shape_list = get_shape_list(self._config)
		# 1. embedding layer
		# 使用了xavier 初始化函数
		emb_table = tf.get_variable(name='embedding',
		                shape=[self._config.vocab_size, self._config.hidden_size],
		                initializer=tf.contrib.layers.xavier_initializer())
		
		x_emb = tf.nn.embedding_lookup(params=emb_table, ids=self._x_id)
		
		# 2. rnn layer
		cell = tf.contrib.rnn.BasicLSTMCell(self._config.hidden_size)
		cell.zero_state(input_shape_list[0], dtype=tf.float32)
		output_keep_prob = self._config.keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
		lstm_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
		
		# B x S x H
		outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self._x_id,
		                  sequence_length=tf.reduce_sum(input_tensor=self._x_mask, axis=-1, keep_dims=False))
		# 3. project layer
		logits = tf.layers.dense(inputs=outputs, units=self._config.vocab_size, activation=tf.nn.relu, use_bias=True)
		self._softmax = tf.nn.softmax(logits=logits, axis=-1)
		
		return logits
	
	def get_prediction(self):
		
		return self._softmax
	
	def get_loss(self, logits):
		
		# B x S
		loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y_id, logits=logits)
		loss_per_sample_after_mask = loss_per_sample * tf.cast(x=self._x_mask, dtype=tf.float32)
		loss = tf.reduce_mean(loss_per_sample)
		self._loss = loss
		