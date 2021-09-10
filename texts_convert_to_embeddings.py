import pandas as pd
from transformers import BertTokenizer,BertConfig
from model.sentence_bert import SentenceBert
import torch
from tqdm import  tqdm
from data_reader.dataReader_nopairs import DataReaderNopairs
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    task_type = "classification"
    # task_type = "regression"
    pretrained = './output/classification/20W_SBert_best_new2021-09-02'
    # pretrained = './output/regression/20W_SBert_best_new_bak_2021-08-31'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 64

    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=pretrained,
                                         max_len=max_len, tokenizer=tokenizer, device=device, task_type=task_type)
    model.to(device)

    with open('data/data_with_tags_2021-09-07-11-51-48.txt','r',encoding='utf-8') as f:
        lines = f.readlines()

    inside_questions = []
    inside_answers = []
    match_ids = []
    for line in lines:
        line = line.strip('\n').split('\t')
        inside_answers.append(line[-1])
        inside_questions.append(line[-2])
        match_ids.append(line[-3].split(':')[-1])

    df = pd.DataFrame()
    df['id'] = match_ids
    df['question'] = inside_questions
    df['answer'] = inside_answers
    writer = pd.ExcelWriter('./output/match_ids_questions_answers_2021_09_02.xlsx')
    df.to_excel(writer,index=False)
    writer.save()



    inside_data = DataReaderNopairs(tokenizer=tokenizer,texts=inside_questions,max_len=64)
    inside_dataloader = DataLoader(dataset=inside_data,shuffle=False,batch_size=64)




    inside_embeddings = embedding(inside_dataloader,model,device)
    embedding_save_path = os.path.join(pretrained,"matchs_embeddings.bin")
    torch.save(inside_embeddings,embedding_save_path)



def embedding(dataloader,model,device):
    vectors = []
    for batch in tqdm(dataloader,desc='embedding'):
        batch = [t.to(device) for t in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
        embedding = model.encoding(inputs)
        vectors.append(embedding)

    vectors = torch.cat(vectors,dim=0)

    return vectors


if __name__ == '__main__':
    main()