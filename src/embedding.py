import torch
from torch import nn

class Embedding():
    def __init__(self, num_embeddings: int, padding_idx: int, embedding_size: int):
        """ Provides the embeddings. 

        Args:
            num_embeddings (int): The number of embeddings.
            padding_idx (int): The index of the padding relation.
            embedding_size (int): The size of the embeddings. 
        """        
      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding =  nn.Embedding(num_embeddings=num_embeddings+1, embedding_dim=embedding_size, padding_idx=padding_idx, dtype=torch.float64, device=self.device, scale_grad_by_freq=True)
        
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        """ The embedding class classifies on call the input.

        Args:
            input (torch.Tensor): An nd array of relations or nodes. 

        Returns:
            torch.Tensor: the embedding of the input
        """
        embedding = self.embedding(input)

        return embedding
       