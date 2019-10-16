import os

from ner.feature_tools import InputExample
import common.bert_tokenizer as tokenization

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""
  
  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()
  
  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()
  
  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()


class NerDataProcessor(DataProcessor):
  
  @classmethod
  def _read_data(cls, input_file):
    """
    Reads a BIO data.
    """
    with open(input_file, 'r') as f:
      lines = []
      words = []
      labels = []
      for line in f:
        contends = line.strip()
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        
        if len(contends) == 0:
          l = ' '.join([label for label in labels if len(label) > 0])
          w = ' '.join([word for word in words if len(word) > 0])
          lines.append([l, w])
          words = []
          labels = []
          continue
        words.append(word)
        labels.append(label)
      return lines
  
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
    # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
    return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
    
  def _create_example(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text=text, label=label))
    return examples
