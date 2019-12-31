from torch.utils.data import DataLoader,Dataset
from transformers import BertModel,BertTokenizer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
import torch
from tqdm import tqdm
import time
import pandas as pd

class SpanClDataset(Dataset):
    def __init__(self,filename,repeat=1):
        self.max_sentence_length = 64
        self.max_spans_num = len(enumerate_spans(range(self.max_sentence_length),max_span_width=3))
        self.repeat = repeat
        self.tokenizer = BertTokenizer.from_pretrained('pretrained_models/Chinese-BERT-wwm/')
        self.data_list = self.read_file(filename)
        self.len = len(self.data_list)
        self.process_data_list = self.process_data()


    def convert_into_indextokens_and_segment_id(self,text):
        tokeniz_text = self.tokenizer.tokenize(text)
        indextokens = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(indextokens)


        pad_indextokens = [0]*(self.max_sentence_length-len(indextokens))
        indextokens.extend(pad_indextokens)
        input_mask_pad = [0]*(self.max_sentence_length-len(input_mask))
        input_mask.extend(input_mask_pad)

        segment_id = [0]*self.max_sentence_length
        return indextokens,segment_id,input_mask

    def get_spans(self,text,traget_word):
        target_spans = []
        for start, end in enumerate_spans(text, max_span_width=3):
            temp = text[start:end+1]
            if temp == traget_word:
                target_spans.append([start, end])
                break
        return target_spans

    def read_file(self,filename):
        data_list = []
        # with open(filename,'r') as f:
        #     lines = f.readlines()
        #     for line in tqdm(lines,desc="加载数据集处理数据集："):
        #         content = line.strip().split('\t')
        #
        #         sentence_a = content[0]
        #
        #         sentence_b = content[1]
        #
        #         label = content[2]
        #         if len(sentence_a)<=self.max_sentence_length and len(sentence_b)<=self.max_sentence_length :
        #             data_list.append((sentence_a,sentence_b,label))
        df = pd.read_csv(filename, sep='\t')  # tsv文件
        s1, s2, labels = df['text_a'], df['text_b'], df['label']

        for sentence_a, sentence_b, label in tqdm(list(zip(s1, s2, labels)),desc="加载数据集处理数据集："):
            if len(sentence_a) <= self.max_sentence_length and len(sentence_b) <= self.max_sentence_length:
                data_list.append((sentence_a, sentence_b, label))
        return data_list

    def process_data(self):
        process_data_list = []
        for ele in tqdm(self.data_list,desc="处理文本信息："):
            res = self.do_process_data(ele)
            process_data_list.append(res)
        return process_data_list

    def do_process_data(self,params):

        res = []
        sentence_a = params[0]
        sentence_b = params[1]
        label = params[2]

        indextokens_a,segment_id_a,input_mask_a = self.convert_into_indextokens_and_segment_id(sentence_a)
        indextokens_a = torch.tensor(indextokens_a,dtype=torch.long)
        segment_id_a = torch.tensor(segment_id_a,dtype=torch.long)
        input_mask_a = torch.tensor(input_mask_a,dtype=torch.long)

        indextokens_b, segment_id_b, input_mask_b = self.convert_into_indextokens_and_segment_id(sentence_b)
        indextokens_b = torch.tensor(indextokens_b, dtype=torch.long)
        segment_id_b = torch.tensor(segment_id_b, dtype=torch.long)
        input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)

        label = torch.tensor(int(label))

        res.append(indextokens_a)
        res.append(segment_id_a)
        res.append(input_mask_a)


        res.append(indextokens_b)
        res.append(segment_id_b)
        res.append(input_mask_b)


        res.append(label)

        return res

    def __getitem__(self, i):
        item = i

        indextokens_a = self.process_data_list[item][0]
        segment_id_a = self.process_data_list[item][1]
        input_mask_a = self.process_data_list[item][2]



        indextokens_b = self.process_data_list[item][3]
        segment_id_b = self.process_data_list[item][4]
        input_mask_b = self.process_data_list[item][5]


        label = self.process_data_list[item][6]


        return indextokens_a,input_mask_a,indextokens_b,input_mask_b,label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.process_data_list)
        return data_len

