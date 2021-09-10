import pandas as pd
from transformers import BertTokenizer,BertConfig
from model.sentence_bert import SentenceBert
import torch
from tqdm import  tqdm
from data_reader.dataReader_nopairs import DataReaderNopairs
from torch.utils.data import DataLoader




def compute_cossim_topk(query_emebdding,inside_embeddings,topn=10):
    d = torch.mul(query_emebdding, inside_embeddings)  # 计算对应元素相乘
    a_len = torch.norm(query_emebdding, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(inside_embeddings, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    topk = torch.topk(cos,topn)
    return topk




def main():
    # task_type = "classification"
    task_type = "regression"
    # pretrained = './output/classification/20W_SBert_best_new2021-08-30'
    pretrained = './output/regression/20W_SBert_best_new_bak_2021-08-31'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 64


    model = SentenceBert.from_pretrained(config=config, pretrained_model_name_or_path=pretrained,
                                         max_len=max_len, tokenizer=tokenizer, device=device, task_type=task_type)

    model.to(device)

    inside_df = pd.read_excel('data/weikong/weikong_template_train_2021-0827_new.xlsx')
    quesitons = inside_df['question'].values.tolist()
    answers = inside_df['answer'].values.tolist()
    inside_questions = []
    inside_answers = []
    for quesiton,answer in zip(quesitons,answers):
        quesiton = quesiton.strip('||').split('||')
        quesiton = list(set(quesiton))
        for ele in quesiton:
            inside_questions.append(ele)
            inside_answers.append(answer)

    inside_data = DataReaderNopairs(tokenizer=tokenizer,texts=inside_questions,max_len=64)
    inside_dataloader = DataLoader(dataset=inside_data,shuffle=False,batch_size=64)

    query_df = pd.read_excel('data/weikong/test_data0824.xlsx')
    query_questions = query_df['question'].values.tolist()
    query_data = DataReaderNopairs(tokenizer=tokenizer, texts=query_questions, max_len=64)
    query_dataloader = DataLoader(dataset=query_data, shuffle=False, batch_size=64)


    inside_embeddings = embedding(inside_dataloader,model,device)
    query_embeddings = embedding(query_dataloader,model,device)

    querys = []
    recommends = []
    simi = []
    recommend_answers = []
    for i in tqdm(range(len(query_questions)),desc='computing similarity'):
        temp = query_embeddings[i:i+1]
        topk = compute_cossim_topk(temp,inside_embeddings,topn=10)
        indexs = topk.indices.data.tolist()
        values = topk.values.data.tolist()

        for index, sim in zip(indexs, values):
            querys.append(query_questions[i])
            recommends.append(inside_questions[index])
            simi.append(sim)
            recommend_answers.append(inside_answers[index])
        querys.append('')
        recommends.append('')
        simi.append('')
        recommend_answers.append('')

    result_df = pd.DataFrame()
    result_df['query'] = querys
    result_df['recommends'] = recommends
    result_df['similarity'] = simi
    result_df['recommend_answer'] = recommend_answers

    # writer = pd.ExcelWriter('output/classification/classification_test_dataset_similarity_20w.xlsx')
    writer = pd.ExcelWriter('output/regression/regression_test_dataset_similarity_20w.xlsx')
    result_df.to_excel(writer,index=False)
    writer.save()



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