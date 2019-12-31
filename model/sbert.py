import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor,SpanExtractor,SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor
import torch
from transformers import BertModel


class SpanBertClassificationModel(nn.Module):
    def __init__(self):
        super(SpanBertClassificationModel,self).__init__()

        self.bert = BertModel.from_pretrained('pretrained_models/Chinese-BERT-wwm/').cuda()
        for param in self.bert.parameters():
            param.requires_grad = True

        # self.hide1 = nn.Linear(768*3,768)
        # self.hide2 = nn.Linear(768,384)
        #
        # self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(768*3,2)

    def forward(self, indextokens_a,input_mask_a,indextokens_b,input_mask_b,mode):
        embedding_a = self.bert(indextokens_a,input_mask_a)[0]
        embedding_b = self.bert(indextokens_b,input_mask_b)[0]

        embedding_a = torch.mean(embedding_a,1)
        embedding_b = torch.mean(embedding_b,1)

        abs = torch.abs(embedding_a - embedding_b)


        target_span_embedding = torch.cat((embedding_a, embedding_b,abs), dim=1)
        # hide_1 = F.relu(self.hide1(target_span_embedding))
        # hide_2 = self.dropout(hide_1)
        # hide = F.relu(self.hide2(hide_2))


        out_put = self.out(target_span_embedding)
        return out_put


