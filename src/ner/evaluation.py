#coding: utf-8

import linecache
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("predict_lable_file", None, "predict result")
flags.DEFINE_string("grund_truth_lable_file", None, 'labels')


class Evaluation:
 
  def __init__(self, predict_lable_file, truth_lable_file):
   
    self._predict_lable_file = predict_lable_file
    self._truth_lable_file = truth_lable_file
 
  def load_prediction(self):
   
    lines = linecache.getlines(self._predict_lable_file)
    tags = list()
    for line in lines:
      line = line.strip()
      tags.append(line)
   
    return tags
 
  def load_truth(self):
   
    lines = linecache.getlines(self._truth_lable_file)
    tags = list()
    tmp_tags = list()
    for line in lines:
      line = line.strip()
      if len(line) == 0:
        if len(tmp_tags) > 126:
          tmp_tags = tmp_tags[:126]
        tags.append('[CLS]')
        for ele in tmp_tags:
          ele_token = ele.split()
          tags.append(ele_token[1])
        tags.append('[SEP]')
        tmp_tags.clear()
      else:
        tmp_tags.append(line)
       
    with open(self._predict_lable_file + ".simplify.txt", 'w') as fp:
      for ele in tags:
        fp.write(ele + "\n")
    return tags
 
  def eval_all(self, names):
   
    num_truth_all = 0
    num_fault_all = 0
    num_tag_all = 0
    for name in names:
      num_truth, num_fault, num_tag = self.eval(name)
      prec, recall, f1 = self.metrice_def(num_truth, num_fault, num_tag)
      tf.logging.info('--------------------------[ %s ]------------------------', name)
      tf.logging.info("prec: %f", prec)
      tf.logging.info("recall: %f", recall)
      tf.logging.info("f1: %f", f1)
     
      num_truth_all += num_truth
      num_fault_all += num_fault
      num_tag_all += num_tag
   
    return self.metrice_def(num_truth_all, num_fault_all, num_tag_all)
 
  def __is_entity_changed(self, tag):
   
    if tag.find('S') == 0 or tag.find('O') == 0 or tag.find('B') == 0:
      return True
    return False
 
  def eval(self, name):
   
    name = '-' + name
    tags_seq_predict = self.load_prediction()
    tags_seq_truth = self.load_truth()
   
    if len(tags_seq_predict) != len(tags_seq_truth):
      raise Exception('格式错误')
   
    l = len(tags_seq_truth)
    num_fault = 0
    num_truth = 0
    num_tag = 0
    i = 0
   
    while i < l:
     
      j = i
      if tags_seq_truth[i] == 'B' + name:
        '''
        BIO可能出现三种种情况：
          1. 完全匹配...BIIO... <->...BIIO... OR ...BMMEO... <->...BMMEO...
          2. 匹配过长...BIIIO...<->...BIIO... 不存在
          3. 匹配错误...IIIO... <->...BIIO... OR ...BMMMO... <->...BMMEO...
        '''
        num_tag += 1
        tag_ = False
        while j < l:
          if j > i and (self.__is_entity_changed(tags_seq_truth[j])):
            break
          if tag_ == False and tags_seq_truth[j] != tags_seq_predict[j]:
            tag_ = True
          j += 1
        if tag_:
          num_fault += 1
        else:
          jj = j
          while j < l and (tags_seq_predict[j] in {'I' + name}):
            j += 1
          if jj > j:
            num_fault += 1
          num_truth += 1
         
      elif tags_seq_truth[i] == 'S' + name:
        '''
        S必须完全匹配
        '''
        num_tag += 1
        if tags_seq_predict[j] == 'S' + name:
          num_truth += 1
        else:
          num_fault += 1
        j += 1
     
      elif tags_seq_truth[i] == 'O':
        tag_BS_num = 0
        while j < l and tags_seq_predict[j] != 'O' and tags_seq_truth[j] == 'O':
          if tags_seq_predict[j] in {'B' + name, 'S' + name}:
            tag_BS_num += 1
          j += 1
        num_fault += tag_BS_num
        j += 1
       
      else:
        j += 1
     
      i = j
   
    return num_truth, num_fault, num_tag
 
  def metrice_def(self, num_truth, num_fault, num_tag):
   
    precision = float(num_truth) / (float(num_fault + num_truth) + 0.01) + 0.000001
    recall = float(num_truth) / (float(num_tag) + 0.01) + 0.000001
    f1 = (2.0 * precision * recall) / (precision + recall)
   
    return precision, recall, f1


def main(_):
  
  tf.logging.set_verbosity(tf.logging.INFO)
  evaluation = Evaluation(predict_lable_file=FLAGS.predict_lable_file, truth_lable_file=FLAGS.truth_lable_file)
 
  prec, recall, f1_score = evaluation.eval_all(['LOC', 'PER', 'ORG'])
 
  tf.logging.info('*********************结果*********************')
  tf.logging.info("precession: %f", prec)
  tf.logging.info("recall: %f", recall)
  tf.logging.info("f1_score: %f", f1_score)


if __name__ == '''__main__''':
  
  tf.app.run()
