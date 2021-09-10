import torch
import argparse
from data_reader.dataReader import DataReader
from model.sentence_bert import SentenceBert
from torch.utils.data import DataLoader

import torch.nn.functional as F
import pandas as pd

from transformers import BertTokenizer,BertConfig
import os
from tools.progressbar import ProgressBar
from tools.log import Logger
from datetime import datetime

logger = Logger('sbert_loger',log_level=10).logger
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_len",type=int,default=64)
    parser.add_argument("--val_file", type=str, default='./output/classification/classification_val_dataset_2w_similarity.xlsx',help="train text file")
    parser.add_argument("--pretrained", type=str, default="./output/classification/20W_SBert_best_new2021-08-30", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--lr", type=int, default=1e-5, help="epochs")
    parser.add_argument("--task_type",type=str,default='classification')
    args = parser.parse_args()
    return args

def predict(args):
    logger.info("args: %s",args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    config = BertConfig.from_pretrained(args.pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_type = args.task_type
    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=args.pretrained,max_len=args.max_len,tokenizer=tokenizer,device=device,task_type=task_type)
    model.to(device)

    val_dataset = DataReader(tokenizer=tokenizer,filepath=args.val_file,max_len=args.max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    acc,predict_labels = valdation(model,val_dataloader,device,task_type)

    print('\n')
    print(acc)

    df = pd.read_excel(args.val_file)

    df['pre_label'] = predict_labels
    writer = pd.ExcelWriter('output/classification/classification_val_dataset_2w_similarity_prelabels.xlsx')
    df.to_excel(writer,index=False)
    writer.save()






def valdation(model,val_dataloader,device,task_type):
    pre_labels = []
    total = 0
    total_correct = 0
    model.eval()
    loss_total = 0
    with torch.no_grad():
        pbar = ProgressBar(n_total=len(val_dataloader), desc='evaldation')
        for step, batch in enumerate(val_dataloader):
            batch = [t.to(device) for t in batch]
            inputs_a = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            inputs_b = {'input_ids': batch[3], 'attention_mask': batch[4], 'token_type_ids': batch[5]}
            labels = batch[6]
            inputs = []
            inputs.append(inputs_a)
            inputs.append(inputs_b)
            output = model(inputs)
            if task_type == "classification":
                pred = torch.argmax(output,dim=1)
                pre_labels.extend(pred.detach().cpu().tolist())
                correct = (labels==pred).sum()
                total_correct += correct
                total += labels.size()[0]
                loss = F.cross_entropy(output,labels)
                pbar(step, {'loss':loss.item()})
            else:
                loss = F.mse_loss(output,labels)
                loss_total += loss.item()
    if task_type == "classification":
        acc = total_correct/total
        return acc,pre_labels
    else:
        return loss_total


def main():
    args =parse_args()
    predict(args)




if __name__ == '__main__':
    main()