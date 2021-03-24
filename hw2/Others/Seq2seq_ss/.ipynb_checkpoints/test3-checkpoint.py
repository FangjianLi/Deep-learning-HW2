from seq_to_seq_test import test_it
from test2 import get_score
import numpy as np
import os

list_1 = np.arange(59,100,5)
if not os.path.exists('score_results.txt'):
    os.mknod('score_results.txt')
    
with open("score_results.txt", "w+") as f:
    for i in list_1:
        try:
            test_it(i)
            f.write("model:{}, score:{} \n".format(i, get_score()))
        except Exception:
            print("some issue")
            pass
            
