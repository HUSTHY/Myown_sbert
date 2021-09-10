import pandas as pd
from transformers import BertTokenizer,BertConfig
from model.sentence_bert import SentenceBert
import torch
from tqdm import  tqdm
from data_reader.dataReader_nopairs import DataReaderNopairs
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def compute_cossim_topk(query_emebdding,inside_embeddings,topn=10):
    d = torch.mul(query_emebdding, inside_embeddings)  # 计算对应元素相乘
    a_len = torch.norm(query_emebdding, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(inside_embeddings, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    topk = torch.topk(cos,topn)
    return topk


def embedding(dataloader,model,device):
    vectors = []
    for batch in tqdm(dataloader,desc='embedding'):
        batch = [t.to(device) for t in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
        embedding = model.encoding(inputs)
        vectors.append(embedding)

    vectors = torch.cat(vectors,dim=0)

    return vectors


def recommend_acc(task_type,pretrained,threshold):
    # task_type = "classification"
    # pretrained = './output/classification/20W_SBert_best_new2021-08-30'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 64

    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=pretrained,
                                         max_len=max_len, tokenizer=tokenizer, device=device, task_type=task_type)

    model.to(device)

    inside_df = pd.read_excel('data/weikong/shanghai_template_inside_2021-0902_new.xlsx')
    inside_df.dropna(inplace=True)
    inside_df.drop_duplicates(inplace=True)
    quesitons = inside_df['question'].values.tolist()
    answers = inside_df['answer'].values.tolist()
    inside_data = DataReaderNopairs(tokenizer=tokenizer, texts=quesitons, max_len=64)
    inside_dataloader = DataLoader(dataset=inside_data, shuffle=False, batch_size=64)
    inside_embeddings = embedding(inside_dataloader, model, device)
    inside_status = []
    correct = 0
    inside_recommend_answers = []
    inside_recommend_questions = []
    inside_sims = []
    for i in range(len(quesitons)):
        temp = inside_embeddings[i:i+1]
        topk = compute_cossim_topk(temp, inside_embeddings, topn=1)
        indexs = topk.indices.data.tolist()
        values = topk.values.data.tolist()

        for index, sim in zip(indexs, values):
            if sim >= threshold:
                inside_recommend_answers.append(answers[index])
                inside_recommend_questions.append(quesitons[index])
                inside_sims.append(sim)
                if answers[index] == answers[i]:
                    correct += 1
                    inside_status.append('正确')
                else:
                    inside_status.append('错误')
            else:
                inside_recommend_answers.append('')
                inside_sims.append(0)
                inside_status.append('错误')
                inside_recommend_questions.append('')

    inside_df['recommend_question'] = inside_recommend_questions
    inside_df['recommend_answer'] = inside_recommend_answers
    inside_df['status'] = inside_status
    inside_df['similarity'] = inside_sims
    acc = correct/len(inside_df)
    print('inside acc:%.4f' % (acc))
    inside_df['accuracy'] = [acc]*len(inside_df)
    writer = pd.ExcelWriter("./output/"+task_type+"inside_2021-0902_result.xlsx")
    inside_df.to_excel(writer,index=False)
    writer.save()



    outside_df = pd.read_excel('data/weikong/shanghai_template_outside_2021-0902_new.xlsx')
    outside_df.dropna(inplace=True)
    outside_df.drop_duplicates(inplace=True)
    outside_quesitons = outside_df['question'].values.tolist()
    outside_answers = outside_df['answer'].values.tolist()
    outside_data = DataReaderNopairs(tokenizer=tokenizer, texts=outside_quesitons, max_len=64)
    outside_dataloader = DataLoader(dataset=outside_data, shuffle=False, batch_size=64)
    outside_embeddings = embedding(outside_dataloader, model, device)
    outside_status = []
    correct = 0
    outside_recommend_answers = []
    outside_recommend_questions = []
    outside_sims = []
    for i in range(len(outside_quesitons)):
        temp = outside_embeddings[i:i + 1]
        topk = compute_cossim_topk(temp, inside_embeddings, topn=1)
        indexs = topk.indices.data.tolist()
        values = topk.values.data.tolist()

        for index, sim in zip(indexs, values):
            if sim >= threshold:
                outside_recommend_answers.append(answers[index])
                outside_sims.append(sim)
                outside_recommend_questions.append(quesitons[index])
                if answers[index] == outside_answers[i]:
                    correct += 1
                    outside_status.append('正确')
                else:
                    outside_status.append('错误')
            else:
                outside_recommend_answers.append('')
                outside_sims.append(0)
                outside_status.append('错误')
                outside_recommend_questions.append('')

    outside_df['recommend_question'] = outside_recommend_questions
    outside_df['recommend_answer'] = outside_recommend_answers
    outside_df['status'] = outside_status
    outside_df['similarity'] = outside_sims
    acc = correct / len(outside_df)
    print('outside acc:%.4f'%(acc))
    outside_df['accuracy'] = [acc] * len(outside_df)
    writer = pd.ExcelWriter("./output/" + task_type + "outside_2021-0902_result.xlsx")
    outside_df.to_excel(writer, index=False)
    writer.save()






if __name__ == '__main__':
    task_type = "classification"
    pretrained = "./output/classification/20W_SBert_best_new2021-09-02"
    threshold = 0.7900
    recommend_acc(task_type,pretrained,threshold)
    # print('*'*100)
    # task_type = "regression"
    # pretrained = "./output/regression/20W_SBert_best_new2021-08-31"
    # threshold = 0.0500
    # recommend_acc(task_type, pretrained, threshold)
