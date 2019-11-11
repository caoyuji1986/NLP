import linecache

def embedding_matrix_load(file_name):
  
  lines = linecache.getlines(file_name)
  m = list()
  for line in lines:
    line = line.strip()
    vec = line.split()
    m.append([float(ele) for ele in vec])
  return m

