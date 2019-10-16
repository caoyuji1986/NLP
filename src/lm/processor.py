import common.bert_tokenizer as tokenization
import os


class InputExample(object):
	
	def __init__(self, guid, sent):
		self.guid = guid
		self.sent = sent


class DataProcessor(object):
	'''
	Base class for data converters for sequence classification data sets.
	'''
	
	def get_train_examples(self, data_dir):
		'''
		Gets a collection of `InputExample`s for the train set.
		'''
		raise NotImplementedError()
	
	def get_dev_examples(self, data_dir):
		'''
		Gets a collection of `InputExample`s for the dev set.
		'''
		raise NotImplementedError()
	
	def get_labels(self):
		'''
		Gets the list of labels for this data set.
		'''
		raise NotImplementedError()


class LMDataProcessor(DataProcessor):
	
	@classmethod
	def _read_data(cls, input_file):
		'''
		Read sentence
		sentence format: [sent1]#_#[sent2]#_#[label]
		return [[sent1, sent2, lable],...]
		'''
		
		with open(input_file, 'r') as f:
			lines = list()
			i = 0
			for line in f:
				line = line.strip()
				lines.append(line)
				i += 1
			return lines
	
	@classmethod
	def _create_example(cls, lines, set_type):
		'''
		create InputExample from [sent1, sent2, lable]
		'''
		examples = []
		for (i, line) in enumerate(lines):
			if i % 100 == 0:
				print(i)
			guid = "%s-%s" % (set_type, i)
			sent = tokenization.convert_to_unicode(line)
			examples.append(InputExample(guid=guid, sent=sent))
		return examples
	
	def get_train_examples(self, data_dir):
		
		return self._create_example(
			self._read_data(os.path.join(data_dir, "train.txt")), "train"
		)
	
	def get_dev_examples(self, data_dir):
		
		return self._create_example(
			self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
		)
	
	def get_test_examples(self, data_dir):
		
		return self._create_example(
			self._read_data(os.path.join(data_dir, "test.txt")), "test"
		)
	
	def get_labels(self):
		
		return ['empty', 'contradiction', 'entailment', 'neutral']

