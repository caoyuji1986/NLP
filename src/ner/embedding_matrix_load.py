import linecache

def embedding_matrix_load(file_name):
  
  lines = linecache.getlines(file_name)
  matrix = list()
  for line in lines:
    line = line.strip()
    vec = line.split()
    matrix.append([float(ele) for ele in vec])
  return matrix

