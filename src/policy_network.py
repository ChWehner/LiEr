import torch
from torch import nn
from lstms import MultiLayerLSTM, LayerNormLSTM, LSTM


class PolicyNetwork(nn.Module):
    """The policy network of the agent"""
    def __init__(self,
                 hidden_size: int,
                 embedding_size: int,
                 mlp_size: int,
                 num_node_embeddings: int,
                 padding_node_key: int,
                 num_relation_embeddings: int,
                 padding_relation_key: int,
                 use_entities: bool=False, 
                 dropout: float=0.1,
                 ):
        """ inits the policy network

        Args:
            hidden_size (int): the hidden size of the lstm. 
            embedding_size (int): the size of the embeddings.
            mlp_size (int): the size of the mlp layers. 
            num_node_embeddings (int) : the number of node embeddings.
            padding_node_key (int): the key of the padding node.
            num_relation_embeddings (int) : the number of relation embeddings.
            padding_relation_key (int): the key of the padding relation.
            use_entities (bool, optional): if entities shall be used as inputs for the forward pass. Defaults to False.
            dropout (float, optional): the dropout rate of the lstm. Defaults to 0.1.
        """
        super(PolicyNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_entities = use_entities

        self.embedding_size = embedding_size
        self.mlp_size = mlp_size
        self.hidden_size = hidden_size

        if self.use_entities:
            self.input_size_factor = 2
        else:
            self.input_size_factor = 1

        self.lstm = MultiLayerLSTM(input_size=self.input_size_factor*self.embedding_size, layer_type=LayerNormLSTM, layer_sizes=(hidden_size,hidden_size,hidden_size), bias=True, dropout=dropout, 
             dropout_method='pytorch')
        self.linear0 = nn.Linear(in_features=(self.input_size_factor*self.embedding_size)+self.hidden_size, out_features=self.mlp_size)
        self.relu0 = nn.LeakyReLU()
        self.dropout0 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=mlp_size, out_features=self.input_size_factor*self.embedding_size)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss(reduction="none")


        self.node_embedding = nn.Embedding(
            num_embeddings=num_node_embeddings,
            embedding_dim=embedding_size,
            padding_idx=padding_node_key,
            dtype=torch.float32,
            device=self.device,
            sparse=True,
        )
        self.relation_embedding = nn.Embedding(
            num_embeddings=num_relation_embeddings,
            embedding_dim=embedding_size,
            padding_idx=padding_relation_key,
            dtype=torch.float32,
            device=self.device,
            sparse=True,
        )

    def a_matrix_forward(self, available_actions: torch.Tensor) -> torch.Tensor:
        """Generates the a matrix (available action matrix). 

        Args:
            available_actions (torch.Tensor): Available actions by instance. 

        Returns:
            torch.Tensor: Stacked embeddings of available actions. 
        """

        relations_embedding = self.relation_embedding(available_actions[:,:,0])    
        embedding = relations_embedding

        if self.use_entities:
            tails_embedding = self.node_embedding(available_actions[:,:,1])             
            embedding = torch.cat((embedding, tails_embedding), 2) 

        return embedding

    def forward(self, actions: torch.Tensor, states: torch.Tensor, available_actions: torch.Tensor, padding_mask:  
                torch.Tensor, hidden_state: torch.Tensor = None, missing_relation_embeddings: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :
        """ Differentiable forward pass of the history-depended policy network. 

        Args:
            actions (torch.Tensor): shape (batch_size, max_degree, 2)
            states (torch.Tensor): shape (batch_size, max_degree, 3)
            available_actions (torch.Tensor): shape (batch_size, max_degree, 2)
            padding_mask (torch.Tensor): shape (batch_size, max_degree, )
            hidden_state (torch.Tensor, optional): The previous hidden states. Is initialized if not provided. Defaults to None.
            missing_relation_embeddings (torch.Tensor, optional): The embedding of the missing relation; shape (batch_size, embedding_size). Is initialized if not provided. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: logits, loss, actions, hidden_state, missing_relation_embeddings
        """               
        previous_actions = self.relation_embedding(actions[:,0]).unsqueeze(dim=0)
        if self.use_entities:
            current_nodes = self.node_embedding(states[:,0]).unsqueeze(dim=0)
            previous_actions = torch.cat((previous_actions, current_nodes), dim = 2)

        a_matrix = self.a_matrix_forward(available_actions)  # A is stacking of avalable actions x 2*hiddenlayers

        # init hidden_state and missing_relations if it is the first forward pass
        if not hidden_state and not missing_relation_embeddings:
            batch_size = available_actions.size(0)
            hidden_state = self.lstm.create_hiddens(batch_size)
            missing_relation_embeddings = self.relation_embedding(states[:,2]).unsqueeze(dim=0) 

        state, hidden_state = self.lstm(previous_actions, hidden_state)

        if self.use_entities:
            current_nodes = self.node_embedding(states[:, 0]).unsqueeze(dim=0)
            state = torch.cat((state, current_nodes), dim = 2)

        state = torch.cat((state, missing_relation_embeddings), dim=2)

        x = torch.squeeze(state, dim=0)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu1(x) 
        x = torch.einsum('ijk,ik->ij', a_matrix ,x) 

        # make every unavailabel action highly negative. mask paddings is True where real actions are available and False where padding actions are available
        x = torch.where(padding_mask, x, -99999.) 
        # calculate sparse softmax cross entropy with logits as loss with logits=x (x pre log softmax and labels=selected action index )
        logits = self.log_softmax(x)
        # choose action randomly given the propabilities from the policy network
        actions = torch.multinomial(input=self.softmax(x), num_samples=1)

        # get negative log-likelihood
        loss = self.nll_loss(logits, torch.squeeze(actions))

        return logits, loss, actions, hidden_state, missing_relation_embeddings

    def save(self, name:str) -> None:
        """Saves model with specified name. 

        Args:
            name (str): the name of the model. 
        """
        torch.save(self.state_dict(), f"models/{name}.pt")
        return

    def load(self, name:str) -> object:
        """Load model with specified name. 

        Args:
            name (str): the name of the model. 

        Returns:
            object: the model
        """
        self.load_state_dict(torch.load(f"models/{name}.pt"))
        self.eval()

        return self
