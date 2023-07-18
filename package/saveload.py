import numpy as np


def save(obj, filename, delimeter=','):
    # saving numpy object of any dimension
    s0 = obj.dtype.__str__()
    s1 = str(obj.shape)[1:-1].replace(', ', delimeter)
    arr = obj.reshape(np.prod(obj.shape))
    s2 = ''
    for i in range(len(arr)):
        s2 += str(arr[i])
        if i < len(arr) - 1:
            s2 += delimeter
    with open(filename, 'w') as file:
        file.write(s0 + '\n' + s1 + '\n' + s2)


def load(filename, delimeter=','):
    # load numpy object saved by previous function
    with open(filename, 'r') as file:
        obj = file.read()
    dtype = np.dtype(obj[:obj.find('\n')])
    obj = obj[obj.find('\n') + 1:]
    shape = np.array(obj[:obj.find('\n')].split(delimeter), dtype=int)
    obj = np.array(obj[obj.find('\n') + 1:].split(delimeter), dtype=dtype).reshape(shape)
    return obj