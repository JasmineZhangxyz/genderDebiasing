"""
Custum Word2Vec

For Zhao et al.'s and Savani et al.'s debiasing methods, we created our own model and skip-gram traning loops below.
"""

#imports
import torch
from torch import nn, optim, sigmoid
import tensorflow
from keras.preprocessing.sequence import skipgrams 
import matplotlib.pyplot as plt

#check which device pytorch will use, set default tensor type to cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
torch.set_default_tensor_type('torch.cuda.FloatTensor') #run on google colab


class skipgram(nn.Module):
    """
    defines the layers of the word2vec model
    
    Embedding Layer Target - target words to compare the context words (output embeddings)
    Embedding Layer Context - context words to compare to target words
    Linear - after the dot product of target and context layers, this linear layer transforms the output to 1 dim to compare with 1 = relevant pair, 0 - irrelevant pair labels 
    """
    
    def __init__(self, size_vocab, embedding_dim):
        super(skipgram, self).__init__()
        self.embeddings_target =  nn.Embedding(size_vocab+1, embedding_dim, max_norm=1).to(device) #what we care about
        self.embeddings_context = nn.Embedding(size_vocab+1, embedding_dim, max_norm=1).to(device) #used in loss calculation
        self.linear = nn.Linear(embedding_dim,1)
        

    def forward(self, target_tensor, context_tensor): #loss
        embedding_t = self.embeddings_target(target_tensor)
        embedding_c = self.embeddings_context(context_tensor)
        
        return torch.sigmoid(self.linear(torch.mul(embedding_t, embedding_c))).squeeze(1)

class Custom_Word2Vec:
    """
    defines the word2vec model
    
    hyperparameters: 
    - embedding_dim: embedding dimension (default 10)
    - LR: learning rate for optimizer (default 0.01)
    - window_size: window of context words to generate skip-gram pairs (default 10)
    - EPOCHS: number of iterations to run training (default 10)
    """
    
    def __init__(self, sentance_tokens, embedding_dim=10, LR=0.01, window_size=10, EPOCHS=10):
        #hyperparamters
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.lr = LR
        self.epochs = EPOCHS
        
        #data, corpus
        self.sentance_tokens = sentance_tokens
        self.corpus_vocab = self.corpus_vocab()
        self.size_vocab = len(self.corpus_vocab)
        
        #model, loss, optimizer
        self.model = skipgram(self.size_vocab, self.embedding_dim)
        self.loss_fcn = nn.BCELoss() # use binary cross entropy as the loss function
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr) #use stochiastic gradient descent

    
    def corpus_vocab(self):
        """
        define a dictionary where keys are words, and the values are the unique ids of the words
        """

        #count frequency of each word
        vocab_counts = {}
        for sentance in self.sentance_tokens:
            for word in sentance:
                vocab_counts[word] = vocab_counts.get(word, 0) + 1


        #create corpus by assigning unique ids
        i = 1
        corpus_vocab = {}
        for k, v in sorted(vocab_counts.items(), key=lambda item: item[1]):
            corpus_vocab[k] = i
            i+=1

        return corpus_vocab

    def create_target_context_pairs(self):
        """
        generate [(target, context), 1] pairs as positive samples - contextually relevant pair
        and [(target, random), 0] pairs as negative samples - contextually irrelevant pair
        """
        
        #get the word ids from the corpus for all the sentances
        word_ids_datatset = [[self.corpus_vocab[word] for word in sentance] for sentance in self.sentance_tokens]
        
        #generate skipgrams (pairs) for all sentances
        skip_grams = [skipgrams(word_ids, vocabulary_size=self.size_vocab, window_size=self.window_size) for word_ids in word_ids_datatset]
        
        return skip_grams
    
    def embedding(self, word):
        idx = self.corpus_vocab[word]
        embedding = self.model.embeddings_target(torch.Tensor([idx]).long())
        return embedding.detach().cpu().numpy()[0]
    
    def train(self, plot=True):
        
        skip_grams = self.create_target_context_pairs() #get pairs

        losses_epochs = []

        #loop over epochs
        for epoch in range(self.epochs):
            total_loss = 0
            
            #iterate through all target, context pairs
            for pairs, labels in skip_grams:
                # zero the gradients
                self.optimizer.zero_grad()

                # calculate loss 
                sentance_losses = []
                for i in range (len(pairs)): #pairs in a sentance
                    target_tensor = torch.Tensor([pairs[i][0]]).long() #target word
                    context_tensor =  torch.Tensor([pairs[i][1]]).long() #context word (true or random)
                    label = torch.Tensor([labels[i]]).float() # 1- relevant, 0 - irrelevent

                    output = self.model(target_tensor, context_tensor)
                    loss_pair = self.loss_fcn(output,label)
                    sentance_losses.append(loss_pair)


                #loss backward, optimizer step
                if sentance_losses:
                    loss = torch.sum(torch.stack(sentance_losses))
                    loss.backward()
                    total_loss+= loss.item()
                    self.optimizer.step()

            print('Epoch:', epoch+1, ' Training Loss:', total_loss)
            losses_epochs.append(total_loss)
        
        # plot loss over epochs
        if plot:
            epochs = [i+1 for i in range(self.epochs)]
            plt.plot(epochs,losses_epochs)
            plt.title('Loss vs Epochs for Word2Vec Skip-Gram Model')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.show()

"""

How to Use:
- define a Custom_Word2Vec instance with the hyperparamters
- call the .train() function
- acess trained embeddings via .embedding()


Example:
word_2_vec = Custom_Word2Vec([['he', 'was', 'cool'], ['she', 'loved', 'meat'], ['you', 'do', 'nothing']])
word_2_vec.train()
print ("embedding: ", word_2_vec.embedding('he'))

"""