
from tqdm import tqdm
import torch
import pandas as pd

class DataReaderNopairs(object):
    def __init__(self,tokenizer,texts,max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len
        self.dataList = self.datas_to_torachTensor()
        self.allLength = len(self.dataList)

    def convert_text2ids(self,text):
        text = text[0:self.max_len-2]
        inputs = self.tokenizer(text)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        paddings = [0] * (self.max_len - len(input_ids))
        input_ids += paddings
        attention_mask += paddings

        token_type_id = [0] * self.max_len

        return input_ids, attention_mask, token_type_id


    def datas_to_torachTensor(self):
        res = []
        for line_a in tqdm(self.texts,desc='tokenization',ncols=50):
            temp = []
            input_ids_a, attention_mask_a, token_type_id_a = self.convert_text2ids(text=line_a)
            input_ids_a = torch.as_tensor(input_ids_a, dtype=torch.long)
            attention_mask_a = torch.as_tensor(attention_mask_a, dtype=torch.long)
            token_type_id_a = torch.as_tensor(token_type_id_a, dtype=torch.long)
            temp.append(input_ids_a)
            temp.append(attention_mask_a)
            temp.append(token_type_id_a)

            # input_ids_b, attention_mask_b, token_type_id_b = self.convert_text2ids(text=line_b)
            # input_ids_b = torch.as_tensor(input_ids_b, dtype=torch.long)
            # attention_mask_b = torch.as_tensor(attention_mask_b, dtype=torch.long)
            # token_type_id_b = torch.as_tensor(token_type_id_b, dtype=torch.long)
            # temp.append(input_ids_b)
            # temp.append(attention_mask_b)
            # temp.append(token_type_id_b)
            #
            # label = torch.as_tensor(label,dtype=torch.long)
            # temp.append(label)

            res.append(temp)
        return res

    def __getitem__(self, item):
        input_ids_a = self.dataList[item][0]
        attention_mask_a = self.dataList[item][1]
        token_type_id_a = self.dataList[item][2]
        return input_ids_a, attention_mask_a, token_type_id_a

        # input_ids_b = self.dataList[item][3]
        # attention_mask_b = self.dataList[item][4]
        # token_type_id_b = self.dataList[item][5]
        #
        # label = self.dataList[item][6]
        
        # return input_ids_a, attention_mask_a, token_type_id_a, input_ids_b, attention_mask_b, token_type_id_b, label


    def __len__(self):
        return self.allLength