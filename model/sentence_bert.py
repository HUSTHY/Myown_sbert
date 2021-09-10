import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers import BertPreTrainedModel
import torch
class SentenceBert(BertPreTrainedModel):
    def __init__(self,config,max_len,tokenizer,device,task_type):
        super(SentenceBert,self).__init__(config)
        self.max_len = max_len
        self.task_type = task_type
        self._target_device = device
        self.tokenizer = tokenizer
        self.bert = BertModel(config=config)
        self.classifier = nn.Linear(3*config.hidden_size,config.num_labels)


    def forward(self,inputs):
        input_a = inputs[0]
        input_b = inputs[1]
        output_a = self.bert(**input_a,return_dict=True, output_hidden_states=True)
        output_b = self.bert(**input_b,return_dict=True, output_hidden_states=True)
        #采用最后一层
        embedding_a = output_a.hidden_states[-1]
        embedding_b = output_b.hidden_states[-1]
        embedding_a = self.pooling(embedding_a,input_a)
        embedding_b = self.pooling(embedding_b, input_b)

        if self.task_type =="classification":
            embedding_abs = torch.abs(embedding_a-embedding_b)
            vectors_concat = []
            vectors_concat.append(embedding_a)
            vectors_concat.append(embedding_b)
            vectors_concat.append(embedding_abs)
            #列拼接3个768————>3*768
            features = torch.cat(vectors_concat, 1)
            output = self.classifier(features)
        else:
            d = torch.mul(embedding_a,embedding_b)
            a_len = torch.norm(embedding_a,dim=1)
            b_len = torch.norm(embedding_b,dim=1)
            cos = torch.sum(d)/(a_len*b_len)
            output = cos
        return output



    def pooling(self,token_embeddings,input):
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        #列拼接
        output_vector = torch.cat(output_vectors, 1)

        return  output_vector



    def encoding(self,inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding


    # def class_infer(self,texts,batch_size=64):
    #     """
    #     推理输入文本list中第一条和剩余其他的是否是同一类别；可以传入无限长的list
    #     :param texts:
    #     :param tokenizer:
    #     :param batch_size:
    #     :return:
    #     """
    #     result = []
    #     text_a = texts[0]
    #     input_id_a, attention_mask_a = self.convert_text2ids(text_a)
    #     input_ids_a = []
    #     attention_masks_a = []
    #     input_ids_b = []
    #     attention_masks_b = []
    #     for text in texts[1:]:
    #         input_id_b, attention_mask_b = self.convert_text2ids(text)
    #
    #         input_ids_a.append(input_id_a)
    #         attention_masks_a.append(attention_mask_a)
    #
    #         input_ids_b.append(input_id_b)
    #         attention_masks_b.append(attention_mask_b)
    #         if len(input_ids_a) >= batch_size:
    #             inputs = []
    #             input_ids_a = torch.as_tensor(input_ids_a,dtype=torch.long,device=self.device)
    #             attention_masks_a = torch.as_tensor(attention_masks_a,dtype=torch.long,device=self.device)
    #             token_type_ids_a = torch.zeros_like(input_ids_a).to(self.device)
    #
    #             inputs_a = {'input_ids': input_ids_a, 'attention_mask': attention_masks_a,'token_type_ids':token_type_ids_a}
    #
    #             input_ids_b = torch.as_tensor(input_ids_b, dtype=torch.long, device=self.device)
    #             attention_masks_b = torch.as_tensor(attention_masks_b, dtype=torch.long, device=self.device)
    #             token_type_ids_b = torch.zeros_like(input_ids_b).to(self.device)
    #
    #             inputs_b = {'input_ids': input_ids_b, 'attention_mask': attention_masks_b,
    #                         'token_type_ids': token_type_ids_b}
    #
    #             inputs.append(inputs_a)
    #             inputs.append(inputs_b)
    #             logits = self.forward(inputs)
    #
    #
    #             lables = torch.argmax(logits)
    #             result.append(lables)
    #
    #
    #             input_ids_a = []
    #             attention_masks_a = []
    #             input_ids_b = []
    #             attention_masks_b = []
    #
    #
    #
    #     inputs = []
    #     input_ids_a = torch.as_tensor(input_ids_a, dtype=torch.long, device=self.device)
    #     attention_masks_a = torch.as_tensor(attention_masks_a, dtype=torch.long, device=self.device)
    #     token_type_ids_a = torch.zeros_like(input_ids_a).to(self.device)
    #
    #     inputs_a = {'input_ids': input_ids_a, 'attention_mask': attention_masks_a, 'token_type_ids': token_type_ids_a}
    #
    #     input_ids_b = torch.as_tensor(input_ids_b, dtype=torch.long, device=self.device)
    #     attention_masks_b = torch.as_tensor(attention_masks_b, dtype=torch.long, device=self.device)
    #     token_type_ids_b = torch.zeros_like(input_ids_b).to(self.device)
    #
    #     inputs_b = {'input_ids': input_ids_b, 'attention_mask': attention_masks_b,
    #                 'token_type_ids': token_type_ids_b}
    #
    #     inputs.append(inputs_a)
    #     inputs.append(inputs_b)
    #     logits = self.forward(inputs)
    #
    #     lables = torch.argmax(logits)
    #     result.append(lables)
    #
    #
    #     return  result



    def class_infer(self,texts,batch_size=64):
        """
        推理输入文本list中第一条和剩余其他的是否是同一类别；传入长度<batch_size
        :param texts:
        :param tokenizer:
        :param batch_size:
        :return:
        """
        assert len(texts)<= batch_size
        result = []
        text_a = texts[0]
        input_id_a, attention_mask_a = self.convert_text2ids(text_a)
        input_ids_a = []
        attention_masks_a = []
        input_ids_b = []
        attention_masks_b = []
        for text in texts[1:]:
            input_id_b, attention_mask_b = self.convert_text2ids(text)

            input_ids_a.append(input_id_a)
            attention_masks_a.append(attention_mask_a)

            input_ids_b.append(input_id_b)
            attention_masks_b.append(attention_mask_b)


        inputs = []
        input_ids_a = torch.as_tensor(input_ids_a, dtype=torch.long, device=self._target_device)
        attention_masks_a = torch.as_tensor(attention_masks_a, dtype=torch.long, device=self._target_device)
        token_type_ids_a = torch.zeros_like(input_ids_a).to(self._target_device)
        inputs_a = {'input_ids': input_ids_a, 'attention_mask': attention_masks_a, 'token_type_ids': token_type_ids_a}


        input_ids_b = torch.as_tensor(input_ids_b, dtype=torch.long, device=self._target_device)
        attention_masks_b = torch.as_tensor(attention_masks_b, dtype=torch.long, device=self._target_device)
        token_type_ids_b = torch.zeros_like(input_ids_b).to(self._target_device)
        inputs_b = {'input_ids': input_ids_b, 'attention_mask': attention_masks_b,
                    'token_type_ids': token_type_ids_b}


        inputs.append(inputs_a)
        inputs.append(inputs_b)


        logits = self.forward(inputs)
        lables = torch.argmax(logits)
        result.append(lables)


        return  result



    def similarity_infer(self,texts,batch_size=64):
        """
        计算输入文本list中第一条和剩余其他文本的相似度 传入长度<batch_size
        :param texts:
        :param tokenizer:
        :param batch_size:
        :return:
        """
        assert len(texts) <= batch_size
        input_ids = []
        attention_masks = []
        for text in texts:
            input_id, attention_mask = self.convert_text2ids(text)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)

        input_ids_a = torch.as_tensor(input_ids, dtype=torch.long, device=self._target_device)
        attention_masks_a = torch.as_tensor(attention_masks, dtype=torch.long, device=self._target_device)
        token_type_ids_a = torch.zeros_like(input_ids_a).to(self._target_device)
        inputs = {'input_ids': input_ids_a, 'attention_mask': attention_masks_a, 'token_type_ids': token_type_ids_a}

        embeddings = self.encoding(inputs)


        embedding_a = embeddings[0:1]
        embedding_b = embeddings[1:]

        d = torch.mul(embedding_a, embedding_b)  # 计算对应元素相乘
        a_len = torch.norm(embedding_a, dim=1)  # 2范数，也就是模长
        b_len = torch.norm(embedding_b, dim=1)
        cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度

        simlaritys = cos

        return simlaritys





    def convert_text2ids(self,text):
        text = text[0:self.max_len - 2]
        inputs = self.tokenizer(text)

        input_ids = inputs['input_ids']
        # lenght = len(input_ids)
        attention_mask = inputs['attention_mask']
        paddings = [0] * (self.max_len - len(input_ids))
        input_ids += paddings
        attention_mask += paddings

        return input_ids, attention_mask
