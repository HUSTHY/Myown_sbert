import torch
import psutil
import gc
if __name__ == '__main__':
    counts = [10000,20000,30000,100000,300000]
    for count in counts:
        mem = psutil.virtual_memory()
        # 系统总计内存(单位字节)
        s_total = float(mem.total)
        # 系统已经使用内存(单位字节)
        s_use = float(mem.used)
        print('s_use:%.4f MB' % (s_use / 1024 / 1024))
        embeddings = torch.randn(count, 768)
        print(embeddings.shape)
        # 系统已经使用内存(单位字节)
        mem = psutil.virtual_memory()
        # 系统总计内存(单位字节)
        e_total = float(mem.total)
        # 系统已经使用内存(单位字节)
        e_use = float(mem.used)
        print('e_use:%.4f MB' % (e_use / 1024 / 1024))
        print('embeddings shape:', embeddings.shape)
        print('embeddings used:%.4f MB' % ((e_use - s_use) / 1024 / 1024))
        del embeddings
        gc.collect()
        print('*'*100)