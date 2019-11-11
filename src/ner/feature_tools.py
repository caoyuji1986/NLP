
import pickle
import collections
import tensorflow as tf

import common.bert_tokenization as tokenization


class InputExample(object):
  
  def __init__(self, guid, text, label=None):
    self.guid = guid
    self.text = text
    self.label = label


class InputFeatures(object):
  
  def __init__(self, input_ids, input_mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


def write_tokens(tokens, output_file, mode):
  if mode == tf.estimator.ModeKeys.PREDICT:
    path = output_file
    wf = open(path, 'a')
    for token in tokens:
      if token != "**NULL**":
        wf.write(token + '\n')
    wf.close()


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode, predict_token_file):
  textlist = example.text.split(' ')
  labellist = example.label.split(' ')
  tokens = []
  labels = []
  
  for i, word in enumerate(textlist):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    label_1 = labellist[i]
    for m in range(len(token)):
      if m == 0:
        labels.append(label_1)
      else:
        labels.append("X")
  # truncate
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[0:(max_seq_length - 2)]
    labels = labels[0:(max_seq_length - 2)]
  
  ntokens = []
  segment_ids = []
  label_ids = []
  
  ntokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])
  
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[labels[i]])
  ntokens.append("[SEP]")
  segment_ids.append(0)
  label_ids.append(label_map["[SEP]"])
  
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(0)
    ntokens.append("**NULL**")
  
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  
  if ex_index < 5:
    tf.logging.info("**##* Example *##**")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
  
  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_ids=label_ids
  )
  write_tokens(tokens=ntokens, mode=mode, output_file=predict_token_file)
  return feature


def file_based_convert_examples_to_features(examples,
                                             label_list,
                                             max_seq_length,
                                             tokenizer,
                                             output_file,
                                             lable2id_file=None,
                                             predict_token_file=None,
                                             mode=None):
  label_map = {}
  for (i, label) in enumerate(label_list, 1):
    label_map[label] = i
  with open(lable2id_file, 'wb') as w:
    pickle.dump(label_map, w)
  
  writer = tf.python_io.TFRecordWriter(output_file)
  for (ex_index, example) in enumerate(examples):
    if ex_index % 500 == 0:
      tf.logging.info("Writing %d of %d" % (ex_index, len(examples)))
    feature = convert_single_example(ex_index, example, label_map, max_seq_length,
                                     tokenizer, mode, predict_token_file)
    
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, mode, drop_remainder):
  name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([seq_length], tf.int64)
  }
  
  def _decode_record(record, name_to_features):
    
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example
  
  def input_fn(params):
    
    d = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    batch_size_key = mode + '_batch_size'
    batch_size = params[batch_size_key]
    tf.logging.info('{} is {}'.format(batch_size_key, batch_size))
    d = d.apply(
      tf.contrib.data.map_and_batch(
        lambda record: _decode_record(record, name_to_features),
        batch_size=batch_size,
        drop_remainder=drop_remainder
      )
    )
    return d
  
  return input_fn

def serving_input_fn_builder(max_seq_length):
  
  def serving_input_fn():
    
    label_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
    
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      'label_ids': label_ids,
      'input_ids': input_ids,
      'input_mask': input_mask,
      'segment_ids': segment_ids,
    })()
    return input_fn
  
  return serving_input_fn

