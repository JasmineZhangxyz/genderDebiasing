"""
Custom Word2Vec

We created our own model and skip-gram traning loops below.
"""

#imports
import torch
from torch import nn, optim, sigmoid
import tensorflow
from keras.preprocessing.sequence import skipgrams 
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import math
import copy

#check which device pytorch will use, set default tensor type to cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
#torch.set_default_tensor_type('torch.cuda.FloatTensor') #to run on google colab


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
    - min_freq: min frequency of word to be present in vocab for easier training (default 100)
    """
    
    def __init__(self, sentance_tokens, embedding_dim=10, LR=0.01, window_size=10, EPOCHS=10, min_freq=100):
        #hyperparamters
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.lr = LR
        self.epochs = EPOCHS
        self.min_freq = min_freq
        
        #data, corpus
        self.sentance_tokens = sentance_tokens
        self.corpus_vocab = self.corpus_vocab()
        self.size_vocab = len(self.corpus_vocab)
        self.skip_grams =  self.create_target_context_pairs()
        
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
        for k, v in sorted(vocab_counts.items(), key=lambda item: item[1], reverse=True):
            if (v < self.min_freq): #break if frequency too low
                break;
            corpus_vocab[k] = i
            i+=1

        return corpus_vocab

    def create_target_context_pairs(self):
        """
        generate [(target, context), 1] pairs as positive samples - contextually relevant pair
        and [(target, random), 0] pairs as negative samples - contextually irrelevant pair
        """
            
        print("Generating Skip Grams...")
        tic = time.perf_counter()
        
        #get the word ids that exist in the corpus for all the sentances
        word_ids_datatset = []
        for sentance in self.sentance_tokens:
            word_ids =[]
            for word in sentance:
                if word in self.corpus_vocab:
                    word_ids.append(self.corpus_vocab[word])
            word_ids_datatset.append(word_ids)
        
        #generate skipgrams (pairs) for all sentances
        skip_grams = [skipgrams(word_ids, vocabulary_size=self.size_vocab, window_size=self.window_size) for word_ids in word_ids_datatset]
        
        toc = time.perf_counter()
        print(f"...({(toc - tic)/60:0.4f}min)")
        
        return skip_grams
    
    def embedding(self, word):
        idx = self.corpus_vocab[word]
        embedding = self.model.embeddings_target(torch.Tensor([idx]).long())
        return embedding.detach().cpu().numpy()[0]
    
    def train(self, plot=True):
        
        #get time estimate for training
        time_finish = datetime.now() + timedelta(seconds=(1/26)*len(self.skip_grams)*self.epochs)
        print("Training. Curr Time =", datetime.now().strftime("%H:%M:%S"), ", Estimated Finish Time =", time_finish.strftime("%H:%M:%S"))
        
        tic = time.perf_counter()
        losses_epochs = []

        #loop over epochs
        for epoch in range(self.epochs):
            tic_e = time.perf_counter()
            total_loss = 0
            
            #iterate through all target, context pairs
            for pairs, labels in self.skip_grams:
                
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
                    
            toc_e = time.perf_counter()
            print(f'Epoch: {epoch+1}, Training Loss: {total_loss}  ({(toc_e - tic_e)/60:0.4f}min)')
            losses_epochs.append(total_loss)
        
        # plot loss over epochs
        if plot:
            epochs = [i+1 for i in range(self.epochs)]
            plt.plot(epochs,losses_epochs)
            plt.title('Loss vs Epochs for Word2Vec Skip-Gram Model')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.show()
        
        toc = time.perf_counter()
        print(f"...({(toc - tic)/60:0.4f}min)")

    # ------------------- Random Pertubation ------------------- 
    # adapted from: https://github.com/abacusai/intraprocessing_debiasing?fbclid=IwAR3TiXq-3idbj1x4IFT4rBDpt5mHTdyYW82k7Ro6se06Etsls06LX0xEjVc
    
    #helpers
    def get_best_thresh(self, threshs, margin, epsilon):
        '''
        calculates best threshold and its corresponding objective function output 
        '''
        objectives = []
        for thresh in threshs:
            objectives.append(self.objective_function(epsilon - margin, thresh))
        return threshs[np.argmax(objectives)], np.max(objectives)

    def compute_performance(self, thresh):
        '''
        Finds y_true (skip gram labels) and y_pred (model outputs with assigned 1 or 0 based on thresh), 
        and returns an accuracy score using balanced_accuracy_score()
        '''
        y_true = []
        y_pred = []

        for pairs, labels in self.skip_grams:
            for i in range (len(pairs)): #pairs in a sentance
                target_tensor = torch.Tensor([pairs[i][0]]).long() #target word
                context_tensor =  torch.Tensor([pairs[i][1]]).long() #context word (true or random)
                y_true.append(labels[i])

                #model output, 1 or 0 based on threshold
                output = self.model(target_tensor, context_tensor)
                output_np = output.detach().numpy()[0]
                if output_np > thresh:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        
        return balanced_accuracy_score(y_true, y_pred)
    
    def objective_function(self, epsilon, thresh):
      bias = 0 #we do not have defined protected groups in this case
      performance = self.compute_performance(thresh)
      return - epsilon*abs(bias) - (1-epsilon)*(1-performance)


    #main 
    def random_debiasing(self,num_trails, stddev, margin, epsilon):
      '''
      Hyperparameters:
        - num_trials - number of iterations
        - stddev: 0.1
        - margin: 0.01
        - epsilon: 0.05
      '''
      rand_result = {'objective': -math.inf, 'model': self.model.state_dict(), 'thresh': -1}

      for iteration in range(num_trails):
          
          for param in self.model.parameters():
              param.data = param.data * (torch.randn_like(param) * stddev + 1)

          threshs = np.linspace(0, 1, 501)
          best_rand_thresh, best_obj = self.get_best_thresh(threshs, margin, epsilon)

          if best_obj > rand_result['objective']:
              rand_result = {'objective': best_obj, 'model': copy.deepcopy(self.model.state_dict()), 'thresh': best_rand_thresh}
        
          print(iteration,"/",num_trails," sampled. Best objective so far: ", rand_result["objective"], "for threshold: ", rand_result["thresh"])

      print('Updating Model with best objective function results.')
      self.model.load_state_dict(rand_result['model']) #load model which had the best objective function

        
        
"""

How to Use:
- define a Custom_Word2Vec instance with the hyperparamters
- call the .train() function
- acess trained embeddings via .embedding()


Example:

word_2_vec = Custom_Word2Vec([['he', 'was', 'cool'], ['she', 'loved', 'meat'], ['you', 'do', 'nothing']], window_size=2, min_freq=1)
#word_2_vec.train()

embedding_before = word_2_vec.embedding('he')

word_2_vec.random_debiasing(10, 0.1, 0.01, 0.05)
embedding_after = word_2_vec.embedding('he')

print(embedding_before, "VS, ", embedding_after)
"""