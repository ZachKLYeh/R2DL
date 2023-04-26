import torch
import torch.nn as nn
import numpy
import numpy as np
from ksvd import ApproximateKSVD
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class Mapped_Embedding_Layer(nn.Module):
    def __init__(self, token_dim, source_embeddings, **kwargs):
        super(Mapped_Embedding_Layer, self).__init__()

        self.source_embeddings = source_embeddings
        self.source_embeddings.requires_grad = False
        self.target_embeddings = torch.zeros([token_dim, source_embeddings.shape[1]])
        self.target_embeddings.requires_grad = False

        self.mapped_coefficient_matrix = nn.Parameter(torch.rand(token_dim, source_embeddings.shape[0]))
        
    def forward(self, token_ids, *args, **kwargs):
        self.target_embeddings = torch.matmul(self.mapped_coefficient_matrix,  self.source_embeddings)
        out = self.target_embeddings[token_ids]
        return out

class R2DL(nn.Module):
    def __init__(self, freeze=True):
        super(R2DL, self).__init__()

        self.transformer = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.source_embeddings = self.transformer.distilbert.embeddings.word_embeddings.weight
        self.mapped_embedding_layer = Mapped_Embedding_Layer(21, self.source_embeddings)
        self.transformer.distilbert.embeddings.word_embeddings = self.mapped_embedding_layer
        
    def reset_parameters(self):
        self.mapped_embedding_layer = Mapped_Embedding_Layer(21, self.source_embeddings)
        self.transformer.distilbert.embeddings.word_embeddings = self.mapped_embedding_layer

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        return output.logits

    def ksvd(self):
        """
        Row-major K-SVD algorithm:
            intput: Y, D(optional)
            output: X, D(updated)
            objective_fn: minimize{norm2(||Y - XD||)} subject to norm0(X) <= k
            task: find the optional dictonay D to repesent source dictionary Y with 
                  Y   = X * D 

            algorithm:
                if D is not given:
                    random_initailze(D)
                for iter in iterations:
                    1. X = OMP.solve(Y = ? * D)
                    2. errorD, errorX = SVD(...)
                    3. D = D - errorD
                    4. X = X - errorX
                    
        R2DL:
            input:  VT, VS(optional)
            output: M , VS(updated)
            replace:
                Y   = X * D
                VT  = M * VS
        """
        target_embedding_layer =  self.transformer.distilbert.embeddings.word_embeddings
        target_embeddings = target_embedding_layer.target_embeddings.clone().detach().cpu().numpy()
        source_embeddings = target_embedding_layer.source_embeddings.clone().detach().cpu().numpy()
        old_coefficient_matrix = target_embedding_layer.mapped_coefficient_matrix.clone().detach().cpu().numpy()

        solver = ApproximateKSVD(n_components=30522, max_iter=1)
        dictionary = solver.fit(target_embeddings, source_embeddings).components_

        new_coefficient_matrix = solver.transform(target_embeddings)
        new_target_embeddings = np.matmul(new_coefficient_matrix, source_embeddings)

        target_embedding_layer.mapped_coefficient_matrix.requires_grad = False
        target_embedding_layer.mapped_coefficient_matrix.copy_(torch.tensor(new_coefficient_matrix))
        target_embedding_layer.mapped_coefficient_matrix.requires_grad = True

        print(f"ksvd_distance:{np.sum(new_target_embeddings-target_embeddings):.3f}")
