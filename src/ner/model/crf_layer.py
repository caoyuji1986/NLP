import tensorflow as tf

def crf_layer(logits, labels, num_labels, seq_len):
	
	trans = tf.get_variable(
		name='tranitions_matrix',
		shape=[num_labels,num_labels],
		initializer=tf.truncated_normal_initializer(stddev=0.02)
	)
	
	log_likelihood, transition = tf.contrib.crf.crg_log_likelihood(
		inputs=logits,
		tag_indices=labels,
		transition_params=trans,
		sequence_length=seq_len
	)
	
	return tf.reduce_mean(-log_likelihood), transition

def crf_decode(potentials, transition_params, sequence_length):
	
	pred_ids, score = tf.contrib.crf.crf_decode(potentials=potentials,
	                                                 transition_params=transition_params,
	                                                 sequence_length=sequence_length)
	return pred_ids, score