import tensorflow as tf
class Gated_IC_Ner:
	
	def __init__(self, x, mask):
		self._x = x
		self._mask = mask
	
	def create_model(self):
		
		cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._config.num_units)
		cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._config.num_units)
		out = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=cell_fw,
			cell_bw=cell_bw,
			sequence_length=tf.reduce_sum(input_tensor=self._mask, axis=-1)
		)