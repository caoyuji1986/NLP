import tensorflow as tf

flags = tf.flags
FLAGS = tf.flags.FLAGS
flags.DEFINE_string("bert_config_file", None, "pre-trained BERT 模型的配置文件")
flags.DEFINE_integer("max_seq_length", 128,"最大序列长度")
flags.DEFINE_string("task_name", "bert-softmax","模型名称")
flags.DEFINE_string("output_dir", None, "checkpoint 的产出位置")
flags.DEFINE_string("data_dir", None, "训练样本的存放位置")
flags.DEFINE_string("vocab_file", None, "BERT 的词表文件")
flags.DEFINE_float("warmup_proportion", 0.1, "启动训练比")
flags.DEFINE_bool("do_eval", True, "")
flags.DEFINE_bool("do_predict", True, "")
flags.DEFINE_bool("do_train", True, "")
flags.DEFINE_integer("train_batch_size", 256, "")
flags.DEFINE_integer("eval_batch_size", 8, "")
flags.DEFINE_integer("infer_batch_size", 8, "")
flags.DEFINE_float("learning_rate", 3e-3, "初始化学习率")
flags.DEFINE_float("num_train_epochs", 20.0, "")
flags.DEFINE_string("init_checkpoint",  None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")
flags.DEFINE_string("export_dir", None, "pb 格式的模型文件的位置， 如果不是None表示要存储模型_")
