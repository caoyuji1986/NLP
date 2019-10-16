import collections
import tensorflow as tf


class InputFeatures(object):
	
	def __init__(self,
	             x_id,
	             x_mask,
	             y_id):
		self.x_id = x_id
		self.x_mask = x_mask
		self.y_id = y_id

def convert_single_example(ex_index,
                           example,
                           tokenizer):
	'''
	Converts a single `InputExample` into a single `InputFeatures`.
	'''
	x_token = ['CLS'] + tokenizer.tokenize(example.sent)
	y_token = x_token[1:] + ['SEP']
	
	x_id = tokenizer.convert_tokens_to_ids(x_token)
	x_mask = [1 for i in range(len(x_id))]
	y_id = tokenizer.convert_tokens_to_ids(y_token)
	
	
	if ex_index < 1:
		tf.logging.info('*** Example ***')
		tf.logging.info('guid: %s' % (example.guid))
		tf.logging.info('x_id: %s' % ' '.join([str(x) for x in x_id]))
		tf.logging.info('x_mask: %s' % ' '.join([str(x) for x in x_mask]))
		tf.logging.info('y_id: %s' % ' '.join([str(y) for y in y_id]))
		
	feature = InputFeatures(
		x_id=x_id,
		x_mask=x_mask,
		y_id=y_id
	)
	return feature


def file_based_convert_examples_to_features(examples,
                                            tokenizer,
                                            output_file
                                            ):
	'''
	Convert a set of `InputExample`s to a TFRecord file.
	'''
	
	writer = tf.python_io.TFRecordWriter(output_file)
	
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info('Writing example %d of %d' % (ex_index, len(examples)))
		
		feature = convert_single_example(ex_index=ex_index,
		                                 example=example,
		                                 tokenizer=tokenizer)
		
		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f
		
		features = collections.OrderedDict()
		features['x_id'] = create_int_feature(feature.x_id)
		features['x_mask'] = create_int_feature(feature.x_mask)
		features['y_id'] = create_int_feature(feature.y_id)
		
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())
	
	writer.close()


def file_based_input_fn_builder(input_file,
                                drop_remainder,
                                mode=tf.estimator.ModeKeys.TRAIN
                                ):
	'''
	Creates an `input_fn` closure to be passed to TPUEstimator.
	'''
	name_to_features = {
		'x_id': tf.VarLenFeature(dtype=tf.int64),
		'x_mask': tf.VarLenFeature(dtype=tf.int64),
		'y_id': tf.VarLenFeature(dtype=tf.int64)
	}
	
	def _decode_record(record, name_to_features):
		'''
		Decodes a record to a TensorFlow example.
		'''
		example = tf.parse_single_example(record, name_to_features)
		
		# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
		# So cast all int64 to int32.
		for name in list(example.keys()):
		
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
				t = tf.sparse_tensor_to_dense(t)
			example[name] = t
		
		return (example['x_id'], example['x_mask']), (example['y_id'])
	
	def input_fn(params):
		'''
		The actual input function.
		'''
		batch_size = params[mode + '_batch_size']
		
		d = tf.data.TFRecordDataset(input_file)
		is_training = mode==tf.estimator.ModeKeys.TRAIN
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=1000000)
		
		d = d.map(
			map_func=lambda record: _decode_record(record, name_to_features),
			num_parallel_calls=16
		)
		padded_shapes = (
			([None], [None]),
			([None])
		)
		padding_values = (
			(0, 0),
			(0)
		)
		d = d.padded_batch(batch_size=batch_size,
		                   padded_shapes=padded_shapes,
		                   padding_values=padding_values,
		                   drop_remainder=drop_remainder)
		# 启动数据pipe line
		d = d.prefetch(1)
		return d
	
	return input_fn

