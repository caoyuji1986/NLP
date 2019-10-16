
import copy
import json

import six
import tensorflow as tf

from ner.model import crf_layer


class CNNNerConfig:
  
  def __init__(self,
               idcnn_macro_block_num=3,
               layers=[1,1,2],
               vocab_size=21128,
               embedding_size=768,
               filter_width=128,
               filter_num=5,
               embedding_keep_prob=0.5,
               idcnn_keep_prob=0.5,
               use_dilated=True,
               use_directed=False,
               embedding_fine_tuning=True
               ):
    
    self.idcnn_macro_block_num = idcnn_macro_block_num
    self.layers = layers
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.filter_width = filter_width
    self.filter_num = filter_num
    self.embedding_keep_prob = embedding_keep_prob
    self.idcnn_keep_prob = idcnn_keep_prob
    self.use_dilated = use_dilated
    self.use_directed = use_directed
    self.embedding_fine_tuning = embedding_fine_tuning
  
  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = CNNNerConfig()
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
  

class CNNNer(object):
  
  def __init__(self,
               idcnn_macro_block_num,
               layers,
               vocab_size,
               embedding_size,
               filter_width,
               filter_num,
               embedding_keep_prob,
               idcnn_keep_prob,
               use_dilated,
               use_directed,
               embedding_fine_tuning,
               embedding_matrix,
               tag_num,
               input_ids,
               input_mask,
               labels,
               mode
               ):
    
    self._idcnn_macro_block_num = idcnn_macro_block_num
    self._layers = layers
    self._vocab_size = vocab_size
    self._tag_num = tag_num
    self._embedding_size = embedding_size
    self._filter_width = filter_width
    self._filter_num = filter_num
    self._embedding_keep_prob = embedding_keep_prob
    self._idcnn_keep_prob = idcnn_keep_prob
    self._use_dilated = use_dilated
    self._use_directed = use_directed
    
    self._initializer = tf.truncated_normal_initializer(stddev=0.02)
    
    self._input_ids = input_ids
    self._input_mask = input_mask
    self._labels = labels
    self._mode = mode

    with tf.name_scope('embedding'):
      if embedding_fine_tuning:
        if embedding_matrix is not None:
          self._embedding_matrix = tf.Variable(initial_value=embedding_matrix,
                                               dtype=tf.float32,
                                               name="embeddings")
        else:
          self._embedding_matrix = tf.get_variable(name="embeddings",
                                                   dtype=tf.float32,
                                                   shape=[vocab_size, embedding_size],
                                                   initializer=self._initializer)


      else:
        self._embedding_matrix = tf.constant(value=embedding_matrix,
                                             dtype=tf.float32,
                                             name="embeddings")

    self._input_embedding = None
    self._cnn_output = None
    self._logits = None
    
    
  def create_model(self):
    '''
    create model
    '''
    """
    N -- idcnn_macro_block_num
    B -- batch_size
    S -- seq_len
    E -- emb_size
    F_n -- num_filter
    F_w -- filter_width
    T -- tag_num
    """
    tf.logging.info("\n\n\n================================================")
    if self._use_directed == False:
      tf.logging.info("cnn type %s", 'dilated' if self._use_dilated else 'normal')
    tf.logging.info("cnn type %s", 'directed' if self._use_directed else 'normal')
    tf.logging.info("================================================\n\n\n")
    self._input_embedding = self.__embedding_layer(input_ids=self._input_ids)
    if self._use_directed:
      self._cnn_output = self.__bi_directed_cnn_layer(model_inputs=self._input_embedding)
    else:
      self._cnn_output = self.__cnn_layer(model_inputs=self._input_embedding)
    self._logits = self.__cnn_project_layer(idcnn_output=self._cnn_output)
    return self.__loss(logits=self._logits, labels=self._labels)
    
  def  __embedding_layer(self, input_ids):
    '''
    embedding layer
    Input -- B x S
    Output -- B x S x E
    '''
    with tf.device('/cpu:0'):
      embedding_input = tf.nn.embedding_lookup(params=self._embedding_matrix, ids=input_ids)
      keep_prob = self._embedding_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
      embedding_input = tf.nn.dropout(x=embedding_input, keep_prob=keep_prob)
    
    return embedding_input
  
  def __directed_cnn_layer(self, model_inputs, reverse=False):
    
    if reverse:
      model_inputs = tf.reverse(tensor=model_inputs, axis=[1])
    
    # B x S x E -> B x 1 x (Fn-1+S) x E
    model_inputs = tf.expand_dims(input=model_inputs, axis=1)
    model_inputs = tf.pad(tensor=model_inputs, paddings=[[0,0], [0,0], [self._filter_width - 1, 0], [0,0]])

    
    with tf.variable_scope("dcnn"):
      final_output = list()
      total_width = 0
      shape = [1, self._filter_width, self._embedding_size, self._filter_num]
      filter_weights = tf.get_variable(
        "idcnn_filter",
        shape=shape,
        initializer=self._initializer)
      # B x 1 x (Fn-1+S) x E -> B x 1 x S x F_n
      layer_input = tf.nn.conv2d(input=model_inputs,
                                 filter=filter_weights,
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 name="init_layer")

      # 2. dilated cnn layers
      final_output = list()
      total_width = 0
      for c_t in range(self._idcnn_macro_block_num):
        for layer_index in range(len(self._layers)):
          with tf.variable_scope("atrous_conv_l-%d" % layer_index, reuse=tf.AUTO_REUSE):
            layer_input = tf.pad(tensor=layer_input, paddings=[[0, 0], [0, 0], [self._filter_width - 1, 0], [0, 0]])
            w = tf.get_variable(name="w",
                                shape=[1, self._filter_width, self._filter_num, self._filter_num],
                                initializer=self._initializer)
            b = tf.get_variable(name="b",
                                shape=[self._filter_num])
      
            dilation = 1
            layer_y = tf.nn.atrous_conv2d(value=layer_input,
                                          filters=w,
                                          rate=dilation,
                                          padding="VALID")
            layer_y = tf.nn.bias_add(value=layer_y, bias=b)
            # tf.contrib.layer.layer_norm()
            layer_y = tf.nn.relu(layer_y)
            if layer_index == len(self._layers) - 1:
              final_output.append(layer_y)
              total_width += self._filter_num
            layer_input = layer_y

      y = tf.concat(values=final_output, axis=3)
      
      # B x 1 x S x F_n*N -> B x S x F_n*N
      y = tf.squeeze(input=y, axis=[1])
     
      cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._filter_num)
      y, _ = tf.nn.dynamic_rnn(
        cell,
        y,
        dtype=tf.float32
      )
      # B x S x F_n*N -> B*S x F_n*N
      # y = tf.reshape(y, [-1, total_width])
      # B x S x F_n-> B*S x F_n
      if reverse:
        y = tf.reverse(tensor=y, axis=[1])
      y = tf.reshape(y, [-1, self._filter_num])
      return y
    
  def __bi_directed_cnn_layer(self, model_inputs):
    
    #使用不同的参数
    with tf.variable_scope('dcnn_l2r'):
      y_forward = self.__directed_cnn_layer(model_inputs=model_inputs, reverse=False)
      
    with tf.variable_scope('dcnn_r2l'):
      y_backward = self.__directed_cnn_layer(model_inputs=model_inputs, reverse=True)
    
    #B*S x Fn*2
    y = tf.concat(values=[y_forward, y_backward], axis=-1)

    W = tf.get_variable("W", shape=[self._filter_num * 2, self._filter_num],
                        dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

    b = tf.get_variable("b", shape=[self._filter_num], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    outputs__ = tf.tanh(tf.nn.xw_plus_b(y, W, b))


    # B*S x 2*H -> B*S x H
    hidden = tf.tanh(tf.nn.xw_plus_b(outputs__, W, b))
    return hidden
    

  def __cnn_layer(self, model_inputs):
    '''
    idcnn/cnn layer
    Input -- B x S x E
    Output -- B*S x F_n*N
    '''
    #B x S x E -> B x 1 x S x E
    model_inputs = tf.expand_dims(input=model_inputs, axis=1)
    
    with tf.variable_scope("idcnn"):
      #1. init layer
      shape = [1, self._filter_width, self._embedding_size, self._filter_num]
      filter_weights = tf.get_variable(
        "idcnn_filter",
        shape=shape,
        initializer=self._initializer)
      #B x 1 X S x E -> B x 1 x S x F_n
      layer_input = tf.nn.conv2d(input=model_inputs,
                                filter=filter_weights,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="init_layer")
      
      #2. dilated cnn layers
      final_output = list()
      total_width = 0
      for c_t in range(self._idcnn_macro_block_num):
        for layer_index in range(len(self._layers)):
          layer = self._layers[layer_index]
          dilation = int(layer)
          with tf.variable_scope("atrous_conv_l-%d"% layer_index, reuse=tf.AUTO_REUSE):
            w = tf.get_variable(name="w",
                                shape=[1, self._filter_width, self._filter_num, self._filter_num],
                                initializer=self._initializer)
            b = tf.get_variable(name="b",
                                shape=[self._filter_num])
            if not self._use_dilated:
              dilation = 1
            layer_y = tf.nn.atrous_conv2d(value=layer_input,
                                           filters=w,
                                           rate=dilation,
                                           padding="SAME")
            layer_y = tf.nn.bias_add(value=layer_y, bias=b)
            # tf.contrib.layer.layer_norm()
            layer_y = tf.nn.relu(layer_y)
            if layer_index == len(self._layers) - 1:
              final_output.append(layer_y)
              total_width += self._filter_num
            layer_input = layer_y
      
      y = tf.concat(values=final_output, axis=3)
      keep_prob = self._idcnn_keep_prob if self._mode == tf.estimator.ModeKeys.TRAIN else 1.0
      y = tf.nn.dropout(x=y, keep_prob=keep_prob)
      #B x 1 x S x F_n*N -> B x S x F_n*N
      y = tf.squeeze(input=y, axis=[1])
      #B x S x F_n*N -> B*S x F_n*N
      y = tf.reshape(y, [-1, total_width])
      return y
  
  def __cnn_project_layer(self, idcnn_output):
    '''
    project layer
    Input -- [B*S, F_n*N]
    Output -- [B x S x T]
    '''
    idcnn_hidden_size = idcnn_output.get_shape()[-1].value
    with tf.variable_scope("project"):
      W = tf.get_variable(name='W',
                          shape=[idcnn_hidden_size, self._tag_num],
                          dtype=tf.float32,
                          initializer=self._initializer)
      b = tf.get_variable('b', initializer=tf.constant(0.001, shape=[self._tag_num]))
      predict = tf.nn.xw_plus_b(x=idcnn_output, weights=W, biases=b)
    x_shape = self._input_ids.get_shape().as_list()
    y = tf.reshape(tensor=predict, shape=[-1, x_shape[1], self._tag_num])
    return y

  def __loss(self, logits, labels):
    '''
    loss calculation
    Input: logits -- B x S x T
    Input: labels -- B x S
    Output: loss, prediction, best_score
    '''
    with tf.variable_scope("crf_loss"):
      seq_length = tf.reduce_sum(input_tensor=self._input_mask, axis=-1)
      loss, transition = crf_layer.crf_layer(logits=logits,
                                             labels=labels,
                                             num_labels=self._tag_num,
                                             seq_len=seq_length)
      pred_ids, best_score = crf_layer.crf_decode(potentials=logits,
                                                  transition_params=transition,
                                                  sequence_length=seq_length)
    return loss, pred_ids, best_score

