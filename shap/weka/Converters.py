import numpy as np

def inst_to_np(inst):
    """ Convert instances to a numpy array """
    inst_list = []
    for ins in inst:
        inst_list.append(ins.values)
    return np.array(inst_list)
