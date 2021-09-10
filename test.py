# import pandas as pd
#
# if __name__ == '__main__':
#
#     data = [[1,1,2,5],[555,66,88,9]]
#     columns = ["A","B","C","D"]
#     df = pd.DataFrame(data,columns=columns)
#     print(df)

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

a = torch.randn(5,4)
print(a)
b = a.to('cuda:0')
print(b)