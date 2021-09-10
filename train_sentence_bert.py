import torch
import argparse
from data_reader.dataReader import DataReader
from model.sentence_bert import SentenceBert
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer,BertConfig
import os
from tools.progressbar import ProgressBar
from tools.log import Logger
from datetime import datetime

logger = Logger('sbert_loger',log_level=10).logger
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_len",type=int,default=64)
    parser.add_argument("--train_file", type=str,default='./data/train_val/classification_train_dataset_20W_0831.xlsx', help="train text file")
    parser.add_argument("--val_file", type=str, default='./data/train_val/classification_val_dataset_2W_0831.xlsx',help="val text file")
    parser.add_argument("--pretrained", type=str, default="./pretrain_models/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=int, default=1e-5, help="epochs")
    parser.add_argument("--task_type",type=str,default='classification')
    args = parser.parse_args()
    return args

def train(args):
    logger.info("args: %s",args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    config = BertConfig.from_pretrained(args.pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_type = args.task_type
    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=args.pretrained,
                                         max_len=args.max_len, tokenizer=tokenizer, device=device, task_type=task_type)
    model.to(device)


    train_dataset = DataReader(tokenizer=tokenizer,filepath=args.train_file,max_len=args.max_len,task_type=task_type)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    val_dataset = DataReader(tokenizer=tokenizer,filepath=args.val_file,max_len=args.max_len,task_type=task_type)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.5, patience=2)
    re_scheduler = ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5, patience=2)

    model.train()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.epochs)
    best_acc = 0.0
    re_loss_min = 1000000.0
    for epoch in range(args.epochs):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step,batch in enumerate(train_dataloader):
            batch = [t.to(device) for t in batch]
            inputs_a = {'input_ids':batch[0],'attention_mask':batch[1],'token_type_ids':batch[2]}
            inputs_b = {'input_ids': batch[3], 'attention_mask': batch[4], 'token_type_ids': batch[5]}
            labels = batch[6]
            inputs = []
            inputs.append(inputs_a)
            inputs.append(inputs_b)
            output = model(inputs)
            if task_type == "classification":
                loss = F.cross_entropy(output,labels)
            else:
                loss = F.mse_loss(output,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar(step, {'loss':loss.item()})


        time_srt = datetime.now().strftime('%Y-%m-%d')
        if task_type == "classification":
            # train_acc = valdation(model,train_dataloader,device,task_type)
            val_acc = valdation(model,val_dataloader,device,task_type)
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(args.model_out,"classification","20W_SBert_best_new"+time_srt)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info("save model")
                model.save_pretrained(save_path)
                tokenizer.save_vocabulary(save_path)
            # logger.info("train_acc: %.4f------val_acc:%.4f------best_acc:%.4f"%(train_acc,val_acc,best_acc))
            logger.info("val_acc:%.4f------best_acc:%.4f" % (val_acc, best_acc))
        else:
            # re_train_loss = valdation(model, train_dataloader, device, task_type)
            re_val_loss = valdation(model, val_dataloader, device, task_type)
            re_scheduler.step(re_val_loss)

            if re_loss_min > re_val_loss:
                re_loss_min = re_val_loss
                save_path = os.path.join(args.model_out, "regression", "20W_SBert_best_new_bak_"+time_srt)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info("save model")
                model.save_pretrained(save_path)
                tokenizer.save_vocabulary(save_path)

            # logger.info("re_train_loss:%.4f------re_val_loss: %.4f------re_loss_min:%.4f" % (re_train_loss,re_val_loss, re_loss_min))
            logger.info("re_val_loss: %.4f------re_loss_min:%.4f" % (re_val_loss, re_loss_min))



def valdation(model,val_dataloader,device,task_type):
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
        return acc
    else:
        return loss_total


def main():
    args =parse_args()
    train(args)




if __name__ == '__main__':
    main()