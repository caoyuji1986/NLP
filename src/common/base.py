# coding: utf8
import six
import tensorflow as tf
from tensorflow.python.eager import context


def layer_normalize(inputs, epsilon=1e-8, scope="layer_normalize"):
  '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
  inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
  scope: Optional scope for `variable_scope`.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
  
  return outputs


def label_smoothing(inputs, epsilon=0.1):
  '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
  inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
  epsilon: Smoothing rate.

  For example,

  ```
  import tensorflow as tf
  inputs = tf.convert_to_tensor([[[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],
    [[1, 0, 0],
     [1, 0, 0],
     [0, 1, 0]]], tf.float32)

  outputs = label_smoothing(inputs)

  with tf.Session() as sess:
      print(sess.run([outputs]))

  >>
  [array([[[ 0.03333334,  0.03333334,  0.93333334],
      [ 0.03333334,  0.93333334,  0.03333334],
      [ 0.93333334,  0.03333334,  0.03333334]],
     [[ 0.93333334,  0.03333334,  0.03333334],
      [ 0.93333334,  0.03333334,  0.03333334],
      [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
  ```
  '''
  V = inputs.get_shape().as_list()[-1]  # number of channels
  return ((1 - epsilon) * inputs) + (epsilon / V)

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if not context.in_eager_mode():
    if name is None:
      name = tensor.name
    if expected_rank is not None:
      assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.
  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))