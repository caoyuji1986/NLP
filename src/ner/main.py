import os
import tensorflow as tf

from common.bert_optimizer import create_optimizer
from common.bert_tokenizer import FullTokenizer
from ner import tf_metrics, embedding_matrix_load
from ner.model.cnn_ner import CNNNerConfig, CNNNer
from ner.feature_tools import file_based_convert_examples_to_features, file_based_input_fn_builder, \
	serving_input_fn_builder
from ner.flag_center import FLAGS
from ner.processor import NerDataProcessor
from ner.model.rnn_ner import RNNNerConfig, UniRNNNer, BiRNNNer


def load_nn_config(nn_type, nn_config_file):
	
	inst = {
		'cnn':    CNNNerConfig.from_json_file,
		'unirnn': RNNNerConfig.from_json_file,
		'birnn':  RNNNerConfig.from_json_file
	}
	inst[nn_type](nn_config_file)
	
def create_cnn_model(config,
                 mode,
                 input_ids,
                 input_mask,
                 labels,
                 tag_num,
                 embedding_matrix):
	
	return CNNNer(
		config,
		mode,
		input_ids,
		input_mask,
		labels,
		tag_num,
		embedding_matrix
	).create_model()

def create_rnn_model(config,
                 mode,
                 input_ids,
                 input_mask,
                 labels,
                 tag_num,
                 embedding_matrix,
                 nn_type):
	
	inst = {
		'unirnn': UniRNNNer,
		'birnn': BiRNNNer
	}
	return inst[nn_type](
		config=config,
		mode=mode,
		input_ids=input_ids,
		input_mask=input_mask,
		labels=labels,
		tag_num=tag_num,
		embedding_matrix=embedding_matrix
	).create_model()

def create_model(config,
                 mode,
                 input_ids,
                 input_mask,
                 labels,
                 tag_num,
                 embedding_matrix,
                 nn_type):
	inst = {
		'cnn': create_cnn_model,
		'unirnn': create_rnn_model,
		'birnn': create_rnn_model
	}
	return inst[nn_type](
		config=config,
		mode=mode,
		input_ids=input_ids,
		input_mask=input_mask,
		labels=labels,
		tag_num=tag_num,
		embedding_matrix=embedding_matrix
	)

def model_fn_builder():
	
	def model_fn(features, labels, mode, params):
		
		input_ids = features['input_ids']
		input_mask = features['input_mask']
		label_ids = features['label_ids']
		
		nn_config = params['nn_config']
		tag_num = params['tag_num']
		embedding_matrix = params['embedding_matrix']
		nn_type = params['nn_type']
		init_lr = params["learning_rate"]
		
		num_train_steps = params["num_train_steps"]
		num_warmup_steps = params["num_warmup_steps"]
		
		loss, predict, score = create_model(config=nn_config,
		                                    mode=mode,
		                                    input_ids=input_ids,
		                                    input_mask=input_mask,
		                                    labels=label_ids,
		                                    tag_num=tag_num,
		                                    embedding_matrix=embedding_matrix,
		                                    nn_type=nn_type)
		
		tvars = tf.trainable_variables()
		for var in tvars:
			tf.logging.info("name=%s #_# shape=%s", var.name, var.shape)
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			logging_hook = tf.train.LoggingTensorHook({"total_loss": loss}, every_n_iter=10)
			train_op = create_optimizer(loss=loss,
			                            init_lr=init_lr,
			                            num_train_steps=num_train_steps,
			                            num_warmup_steps=num_warmup_steps,
			                            use_tpu=False)
			output_spec = tf.estimator.EstimatorSpec(mode=mode,
			                                         loss=loss,
			                                         train_op=train_op,
			                                         training_hooks=[logging_hook])
			return output_spec
		
		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(label_ids, predictions):
				interest_label_ids = [id for id in range(2, tag_num - 1)]
				
				prec = tf_metrics.precision(labels=label_ids,
				                            predictions=predictions,
				                            num_classes=tag_num,
				                            pos_indices=interest_label_ids,
				                            average='macro')
				
				recall = tf_metrics.recall(labels=label_ids,
				                            predictions=predictions,
				                            num_classes=tag_num,
				                            pos_indices=interest_label_ids,
				                            average='macro')
				
				f1 = tf_metrics.f1(labels=label_ids,
				                   predictions=predictions,
				                   num_classes=tag_num,
				                   pos_indices=interest_label_ids,
				                   average='macro')
				
				return {
					"prec": prec,
					"recall": recall,
					"f1": f1
				}
			
			eval_metrics = metric_fn(label_ids=label_ids, predictions=predict)
			output_spec = tf.estimator.EstimatorSpec(
				mode=mode,
				loss=loss,
				eval_metric_ops=eval_metrics
			)
			return output_spec
		
		else:
			export_outputs = {
				"y": tf.estimator.export.PredictOutput({
					"predict": predict
				})
			}
			output_spec = tf.estimator.EstimatorSpec(predictions=predict, export_outputs=export_outputs)
			return output_spec
	return model_fn
		
def main(_):

	nn_config = load_nn_config(FLAGS.nn_type)
	
	# 使用pretrain embedding
	embedding_matrix = None
	if FLAGS.embedding_matrix_file is not None and tf.gfile.Exists(FLAGS.embedding_matrix_file):
		embedding_matrix = embedding_matrix_load.embedding_matrix_load(FLAGS.embedding_matrix_file)
	model_fn = model_fn_builder()
	processor = NerDataProcessor()
	run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir, save_checkpoints_step=FLAGS.save_checkpoints_step)
	estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={
		'nn_config':nn_config,
		'tag_num': len(processor.get_labels()) + 1,
		'embedding_matrix':embedding_matrix,
		'nn_type': FLAGS.nn_type,
		"learning_rate": FLAGS.learning_rate,
		"num_train_steps": FLAGS.num_train_steps,
		"num_warmup_steps": FLAGS.num_warmup_steps
	})

	tokenizer = FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
	l2id_file = os.path.join(FLAGS.data_dir, 'l2id.txt')
	if FLAGS.do_train:

		train_examples = processor.get_train_examples(FLAGS.data_dir)
		train_steps = len(train_examples) * FLAGS.epoch_num / FLAGS.batch_size
		train_file = os.path.join(FLAGS.data_dir, 'train.tf_record')
		labels = processor.get_labels()
		file_based_convert_examples_to_features(
			examples=train_examples,
			label_list=labels,
			max_seq_length=FLAGS.max_seq_length,
			tokenizer=tokenizer,
			output_file=train_file,
			lable2id_file=l2id_file,
			mode=tf.estimator.ModeKeys.TRAIN
		)
		train_input_fn = file_based_input_fn_builder(
			input_file=train_file,
			seq_length=FLAGS.max_seq_length,
			mode=tf.estimator.ModeKeys.TRAIN,
			drop_remainder=False
		)
		estimator.train(input_fn=train_input_fn, max_steps=train_steps)

	if FLAGS.do_eval:
		
		eval_examples = processor.get_dev_examples(FLAGS.data_dir)
		eval_steps = len(eval_examples) * 1 / FLAGS.batch_size
		eval_file = os.path.join(FLAGS.data_dir, 'eval.tf_record')
		labels = processor.get_labels()
		
		file_based_convert_examples_to_features(
			examples=eval_examples,
			label_list=labels,
			max_seq_length=FLAGS.max_seq_length,
			tokenizer=tokenizer,
			output_file=eval_file,
			lable2id_file=l2id_file,
			mode=tf.estimator.ModeKeys.EVAL
		)
		eval_input_fn = file_based_input_fn_builder(
			input_file=eval_file,
			seq_length=FLAGS.max_seq_length,
			mode=tf.estimator.ModeKeys.EVAL,
			drop_remainder=False
		)
		result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
		output_eval_file = os.path.join(FLAGS.output_dir, 'token_eval.txt')
		with open(output_eval_file, mode='w') as writer:
			for key in sorted(result.keys()):
				writer.write("%s = %s" % (key, result[key]))

	if FLAGS.do_predict:
		
		predict_examples = processor.get_test_examples(FLAGS.data_dir)
		predict_steps = len(predict_examples) * 1 / FLAGS.batch_size
		predict_file = os.path.join(FLAGS.data_dir, 'predict.tf_record')
		labels = processor.get_labels()
		
		file_based_convert_examples_to_features(
			examples=predict_examples,
			label_list=labels,
			max_seq_length=FLAGS.max_seq_length,
			tokenizer=tokenizer,
			output_file=predict_file,
			lable2id_file=l2id_file,
			mode=tf.estimator.ModeKeys.EVAL
		)
		predict_input_fn = file_based_input_fn_builder(
			input_file=predict_file,
			seq_length=FLAGS.max_seq_length,
			mode=tf.estimator.ModeKeys.EVAL,
			drop_remainder=False
		)
		result = estimator.evaluate(input_fn=predict_input_fn, steps=predict_steps)
		output_file = os.path.join(FLAGS.output_dir, 'token_predict.txt')
		with open(output_file, mode='w') as writer:
			for ele in result:
				output_lines = "\n".join(labels[id] for id in ele if id != 0)
				writer.write(output_lines)
	
	if FLAGS.export_dir is not None:
		estimator._export_to_tpu = False
		serving_input_fn = serving_input_fn_builder(max_seq_length=FLAGS.max_seq_length)
		estimator.export_saved_model(FLAGS.export_dir, serving_input_fn)


if __name__=='''__main__''':
	
	tf.app.run()