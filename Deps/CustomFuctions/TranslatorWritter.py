import torch
import numpy as np
from datetime import datetime
import os
import scipy.io as io

# open file and record derived contexts
def  context_recd(path):
    testlog = open(path + '/test_context.md', mode='a', encoding='utf-8')
    print('epoch: %d' % (epoch), file=testlog)
    print('true contexts for net2:', file=testlog)
    print(targets, file=testlog)
    print('predicted contexts for net2:', file=testlog)
    print(output, file=testlog)
    print('Loss: %.3e' % (test_loss), file=testlog)
    testlog.close()