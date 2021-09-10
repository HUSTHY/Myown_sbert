import pandas as pd
from transformers import BertTokenizer,BertConfig
from model.sentence_bert import SentenceBert
import torch
from tqdm import  tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def  similarity_valdation():
    task_type = "classification"
    pretrained = './output/classification/20W_SBert_best_new2021-09-02'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 64
    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=pretrained,
                                         max_len=max_len, tokenizer=tokenizer, device=device, task_type=task_type)
    model.to(device)

    df = pd.read_excel('./data/train_val/classification_val_dataset_2W_0831.xlsx')
    texts_a = df['text_a'].values.tolist()
    texts_b = df['text_b'].values.tolist()
    simlaritys = []
    for a, b in tqdm(zip(texts_a, texts_b), desc='compute similarity'):
        sim = model.similarity_infer([a, b])
        sim = sim.detach().cpu().tolist()[0]
        simlaritys.append(sim)

    df['pre_classification_similarity'] = simlaritys
    writer = pd.ExcelWriter('./output/classification/classification_val_dataset_2W_0831_similarity.xlsx')
    df.to_excel(writer, index=False)
    writer.save()


def comutpe_sim():
    task_type = "classification"
    pretrained = './output/classification/20W_SBert_best_new2021-08-30'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 64
    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=pretrained, max_len=max_len,
                         tokenizer=tokenizer, device=device, task_type=task_type)
    model.to(device)
    a = '再给我申请一点东西'
    b = '我说如果今年出险明年保费还不懂'
    sim = model.similarity_infer([a, b])
    sim = sim.detach().cpu().tolist()[0]
    print(sim)



if __name__ == '__main__':
    similarity_valdation()
    # comutpe_sim()