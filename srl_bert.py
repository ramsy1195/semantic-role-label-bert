#!/usr/bin/env python
# coding: utf-8

# # Semantic Role Labelling with BERT
# 
# The goal of this project is to train and evaluate a PropBank-style semantic role labeling (SRL) system. Following (Collobert et al. 2011) and others, we will treat this problem as a sequence-labeling task. For each input token, the system will predict a B-I-O tag, as illustrated in the following example:
# 
# |The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
# |---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|             
# |B-ARG1|I-ARG1|B-V|B-ARG2|I-ARG2|I-ARG2|I-ARG2|I-ARG2|O|O|O|O|O|O|O|
# |||schedule.01|||||||||||||
# 
# Note that the same sentence may have multiple annotations for different predicates
# 
# |The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
# |---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|             
# |B-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|O|B-V|B-ARG2|I-ARG2|I-ARG2|B-ARGM-TMP|O|
# ||||||||||remove.01||||||
# 
# and not all predicates need to be verbs
# 
# |The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
# |---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|    
# |O|O|O|O|O|O|B-ARG1|B-V|O|O|O|O|O|O|O|
# ||||||||try.02||||||||
# 
# The SRL system will be implemented in [PyTorch](https://pytorch.org/). We will use BERT (in the implementation provided by the [Huggingface transformers](https://huggingface.co/docs/transformers/index) library) to compute contextualized token representations and a custom classification head to predict semantic roles. We will fine-tune the pretrained BERT model on the SRL task.
# 

# ### Overview of the Approach
# 
# The model we will train is pretty straightforward. Essentially, we will just encode the sentence with BERT, then take the contextualized embedding for each token and feed it into a classifier to predict the corresponding tag.
# 
# Because we are only working on argument identification and labeling (not predicate identification), it is essentially that we tell the model where the predicate is. This can be accomplished in various ways. The approach we will choose here repurposes Bert's *segment embeddings*.
# 
# Recall that BERT is trained on two input sentences, seperated by [SEP], and on a next-sentence-prediction objective (in addition to the masked LM objective). To help BERT comprehend which sentence a given token belongs to, the original BERT uses a segment embedding, using A for the first sentene, and B for the second sentence 2.
# Because we are labeling only a single sentence at a time, we can use the segment embeddings to indicate the predicate position instead: The predicate is labeled as segment B (1) and all other tokens will be labeled as segment A (0).
# 
# <img src="https://github.com/daniel-bauer/4705-f23-hw5/blob/main/bert_srl_model.png?raw=true" width=400px>

# ## Setup: GCP, Jupyter, PyTorch, GPU
# 
# To make sure that PyTorch is available and can use the GPU, we run the following cell which should return True. If it doesn't, we should recheck if the GPU drivers and CUDA are installed correctly.
# 
# Note: GPU support is required for this project.

# In[2]:


import torch
torch.cuda.is_available()


# ## Dataset: Ontonotes 5.0 English SRL annotations
# 
# We will work with the English part of the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) data. This is an extension of PropBank, using the same type of annotation. Ontonotes contains annotations other than predicate/argument structures, but we will use the PropBank style SRL annotations only. *Important*: The data set used in this project was provided to me for use by my university. My university is a subscriber to LDC and is allowed to use the data for educational purposes. However, the dataset may not be used in projects unrelated to teaching or research conducted at my university. Hence the following cell where the data source link is provided for download is hidden from view. The data is downloaded in ontonotes_srl.zip.
# 
# 
# 

# In[4]:


get_ipython().system(' unzip ontonotes_srl.zip')


# The data has been pre-processed in the following format. There are three files:
# 
# `propbank_dev.tsv`	`propbank_test.tsv`	`propbank_train.tsv`
# 
# Each of these files is in a tab-separated value format. A single predicate/argument structure annotation consists of four rows. For example
# 
# ```
# ontonotes/bc/cnn/00/cnn_0000.152.1
# The     judge   scheduled       to      preside over    his     trial   was     removed from    the     case    today   /.
#                 schedule.01
# B-ARG1  I-ARG1  B-V     B-ARG2  I-ARG2  I-ARG2  I-ARG2  I-ARG2  O       O       O       O       O       O       O
# ```
# 
# * The first row is a unique identifier (1st annotation of the 152nd sentence in the file ontonotes/bc/cnn/00/cnn_0000).
# * The second row contains the tokens of the sentence (tab-separated).
# * The third row contains the probank frame name for the predicate (empty field for all other tokens).
# * The fourth row contains the B-I-O tag for each token.
# 
# The file `rolelist.txt` contains a list of propbank BIO labels in the dataset (i.e. possible output tokens). This list has been filtered to contain only roles that appeared more than 1000 times in the training data.
# We will load this list and create mappings from numeric ids to BIO tags and back.

# In[5]:


role_to_id = {}
with open("role_list.txt",'r') as f:
    role_list = [x.strip() for x in f.readlines()]
    role_to_id = dict((role, index) for (index, role) in enumerate(role_list))
    role_to_id['[PAD]'] = -100

    id_to_role = dict((index, role) for (role, index) in role_to_id.items())


# Note that we are also mapping the '[PAD]' token to the value -100. This allows the loss function to ignore these tokens during training.

# 
# 
# 

# ## Part 1 - Data Preparation
# 
# Before we can build the SRL model, we first need to preprocess the data.
# 
# 
# ### 1.1 - Tokenization
# 
# One challenge is that the pre-trained BERT model uses subword ("WordPiece") tokenization, but the Ontonotes data does not. Fortunately Huggingface transformers provides a tokenizer.

# In[6]:


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer.tokenize("This is an unbelievably boring test sentence.")


# We need to be able to maintain the correct labels (B-I-O tags) for each of the subwords.
# The following function that takes a list of tokens and a list of B-I-O labels of the same length as parameters, and returns a new token / label pair, as illustrated in the following example.
# 
# 
# ```
# >>> tokenize_with_labels("the fancyful penguin devoured yummy fish .".split(), "B-ARG0 I-ARG0 I-ARG0 B-V B-ARG1 I-ARG1 O".split(), tokenizer)
# (['the',
#   'fancy',
#   '##ful',
#   'penguin',
#   'dev',
#   '##oured',
#   'yu',
#   '##mmy',
#   'fish',
#   '.'],
#  ['B-ARG0',
#   'I-ARG0',
#   'I-ARG0',
#   'I-ARG0',
#   'B-V',
#   'I-V',
#   'B-ARG1',
#   'I-ARG1',
#   'I-ARG1',
#   'O'])
# 
# ```
# 
# To approach this problem, we will iterate through each word/label pair in the sentence and call the tokenizer on the word. This may result in one or more tokens. The, we will create the correct number of labels to match the number of tokens. We have to take care to not generate multiple B- tokens.
# 
# 
# This approach is a bit slower than tokenizing the entire sentence, but is necessary to produce proper input tokenization for the pre-trained BERT model, and the matching target labels.

# In[7]:


def tokenize_with_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
      tokens = tokenizer.tokenize(word)

      for i, token in enumerate(tokens):
          if i == 0 or label == 'O':
              tokenized_sentence.append(token)
              labels.append(label)
          else:
              tokenized_sentence.append(token)
              labels.append('I-' + label[2:])  # Remove 'B-' and add 'I-'

    return tokenized_sentence, labels


# In[8]:


tokenize_with_labels("the fancyful penguin devoured yummy fish .".split(), "B-ARG0 I-ARG0 I-ARG0 B-V B-ARG1 I-ARG1 O".split(), tokenizer)


# ### 1.2 Loading the Dataset
# 
# Next, we are creating a PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class. This class acts as a contained for the training, development, and testing data in memory. 
# 
# 1.2.1 Let us write the \_\_init\_\_(self, filename) method that reads in the data from a data file (specified by the filename).
# 
# For each annotation we start with  the tokens in the sentence, and the BIO tags. Then we create the following
# 
# 1. calling the `tokenize_with_labels` function to tokenize the sentence.
# 2. adding the (token, label) pair to the self.items list.
# 
# 1.2.2 Let us write the \_\_len\_\_(self) method that returns the total number of items.
# 
# 1.2.3 Let us write the \_\_getitem\_\_(self, k) method that returns a single item in a format BERT will understand.
# * We need to process the sentence by adding "\[CLS\]" as the first token and "\[SEP\]" as the last token. The need to pad the token sequence to 128 tokens using the "\[PAD\]" symbol. This needs to happen both for the inputs (sentence token sequence) and outputs (BIO tag sequence).
# * We need to create an *attention mask*, which is a sequence of 128 tokens indicating the actual input symbols (as a 1) and \[PAD\] symbols (as a 0).
# * We need to create a *predicate indicator* mask, which is a sequence of 128 tokens with at most one 1, in the position of the "B-V" tag. All other entries should be 0. The model will use this information to understand where the predicate is located.
# 
# * Finally, we need to convert the token and tag sequence into numeric indices. For the tokens, this can be done using the `tokenizer.convert_tokens_to_ids` method. For the tags, use the `role_to_id` dictionary.
# Each sequence must be a pytorch tensor of shape (1,128). You can convert a list of integer values like this `torch.tensor(token_ids, dtype=torch.long)`.
# 
# To keep everything organized, we will return a dictionary in the following format
# 
# ```
# {'ids': token_tensor,
#  'targets': tag_tensor,
#  'mask': attention_mask_tensor,
#  'pred': predicate_indicator_tensor}
# ```
# 
# 
# (To debug these, we will read in the first annotation only / the first few annotations)
# 

# In[9]:


from torch.utils.data import Dataset, DataLoader

class SrlData(Dataset):

    def __init__(self, filename):
        super(SrlData, self).__init__()

        self.max_len = 128  # the max number of tokens inputted to the transformer.
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.items = []

        # Read the data from the provided file
        with open(filename, 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines), 4):
            sentence_tokens = lines[i + 1].strip().split()
            frame_name = lines[i + 2].strip()
            tags = lines[i + 3].strip().split()

            # Print each sentence's tokenized information for debugging
            #print(f"\nProcessing sentence {i//4 + 1} (line {i+1}):")
            #print("Sentence tokens:", sentence_tokens)
            #print("Frame name:", frame_name)
            #print("Tags:", tags)

            # Tokenize with labels
            tokenized_sentence, tokenized_labels = tokenize_with_labels(sentence_tokens, tags, self.tokenizer)
            #print("Tokenized sentence:", tokenized_sentence)
            #print("Tokenized labels:", tokenized_labels)

            self.items.append((tokenized_sentence, tokenized_labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):

        tokenized_sentence, tokenized_labels = self.items[k]
        tokens = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        labels = ['O'] + tokenized_labels + ['O']

        padding_length = self.max_len - len(tokens)
        if padding_length > 0:
            tokens = tokens + ['[PAD]'] * padding_length
            labels = labels + ['[PAD]'] * padding_length
        else:
            tokens = tokens[:self.max_len]
            labels = labels[:self.max_len]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [role_to_id.get(label, -100) for label in labels]

        attention_mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
        predicate_mask = [0] * self.max_len
        try:
            pred_index = labels.index('B-V')
            predicate_mask[pred_index] = 1
        except ValueError:
            pass

        # Convert the lists to tensors
        token_tensor = torch.tensor(token_ids, dtype=torch.long)  # Shape (128)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)  # Shape (128)
        attn_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)  # Shape (128)
        pred_tensor = torch.tensor(predicate_mask, dtype=torch.long)  # Shape (128)

        return {
            'ids': token_tensor,
            'targets': label_tensor,
            'mask': attn_mask_tensor,
            'pred': pred_tensor
        }



# In[10]:


# Reading the training data takes a while for the entire data because we preprocess all data offline
data = SrlData("propbank_train.tsv")


# ## 2. Model Definition

# In[11]:


from torch.nn import Module, Linear, CrossEntropyLoss
from transformers import BertModel


# We will define the pyTorch model as a subclass of the [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. 

# In[12]:


class SrlModel(Module):

    def __init__(self):

        super(SrlModel, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        # The following two lines would freeze the BERT parameters and allow us to train the classifier by itself.
        # We are fine-tuning the model, so you can leave this commented out!
        # for param in self.encoder.parameters():
        #    param.requires_grad = False

        # The linear classifier head, see model figure in the introduction.
        self.classifier = Linear(768, len(role_to_id))


    def forward(self, input_ids, attn_mask, pred_indicator):

        # This defines the flow of data through the model

        # Note the use of the "token type ids" which represents the segment encoding explained in the introduction.
        # In our segment encoding, 1 indicates the predicate, and 0 indicates everything else.
        bert_output =  self.encoder(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=pred_indicator)

        enc_tokens = bert_output[0] # the result of encoding the input with BERT
        logits = self.classifier(enc_tokens) #feed into the classification layer to produce scores for each tag.

        # Note that we are only interested in the argmax for each token, so we do not have to normalize
        # to a probability distribution using softmax. The CrossEntropyLoss loss function takes this into account.
        # It essentially computes the softmax first and then computes the negative log-likelihood for the target classes.
        return logits


# In[13]:


model = SrlModel().to('cuda') # create new model and store weights in GPU memory


# Now we are ready to try running the model with just a single input example to check if it is working correctly. Clearly it has not been trained, so the output is not what we expect. But we can see what the loss looks like for an initial sanity check.
# 
# Next steps:
# * Taking a single data item from the dev set, as provided by our Dataset class defined above. Obtaining the input token ids, attention mask, predicate indicator mask, and target labels.
# * Running the model on the ids, attention mask, and predicate mask like this:

# In[14]:


# pick an item from the dataset. Then run

devData = SrlData("propbank_dev.tsv")
example = devData[0]

# Extract inputs from the example and remove the extra dimension by using squeeze(0)
ids = example['ids'].to('cuda')  # Shape: (1, 128)
mask = example['mask'].to('cuda')  # Shape: (1, 128)
pred = example['pred'].to('cuda')  # Shape: (1, 128)
targets = example['targets'].to('cuda')  # Shape: (1, 128)

outputs = model(ids.unsqueeze(0), mask.unsqueeze(0), pred.unsqueeze(0))

print("Outputs of the model:", outputs)


# Let us compute the loss on this one item only.
# The initial loss should be close to -ln(1/num_labels)
# 
# Without training we would assume that all labels for each token (including the target label) are equally likely, so the negative log probability for the targets should be approximately $$-\ln(\frac{1}{\text{num_labels}}).$$ This is what the loss function should return on a single example. This is a good sanity check to run for any multi-class prediction problem.

# In[15]:


import math
-math.log(1 / len(role_to_id), math.e)


# In[16]:


loss_function = CrossEntropyLoss(ignore_index = -100, reduction='mean')

num_labels = len(role_to_id)

# Reshape the output to (batch_size * sequence_length, num_labels)
logits = outputs.view(-1, num_labels)

# Reshape the target labels to (batch_size * sequence_length)
targets = targets.view(-1)

# Compute the loss
loss = loss_function(logits, targets)

# Print the loss value
print(f"Computed loss: {loss.item()}")
# complete this. Note that you still have to provide a (batch_size, input_pos)
# tensor for each parameter, where batch_size =1

# outputs = model(ids, mask, pred)
# loss = loss_function(...)
# loss.item()   #this should be approximately the score from the previous cell


# At this point, we should also obtain the actual predictions by taking the argmax over each position.
# The result should look something like this (values will differ).
# 
# ```
# tensor([[ 1,  4,  4,  4,  4,  4,  5, 29, 29, 29,  4, 28,  6, 32, 32, 32, 32, 32,
#          32, 32, 30, 30, 32, 30, 32,  4, 32, 32, 30,  4, 49,  4, 49, 32, 30,  4,
#          32,  4, 32, 32,  4,  2,  4,  4, 32,  4, 32, 32, 32, 32, 30, 32, 32, 30,
#          32,  4,  4, 49,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6, 32, 32,
#          30, 32, 32, 32, 32, 32, 30, 30, 30, 32, 30, 49, 49, 32, 32, 30,  4,  4,
#           4,  4, 29,  4,  4,  4,  4,  4,  4, 32,  4,  4,  4, 32,  4, 30,  4, 32,
#          30,  4, 32,  4,  4,  4,  4,  4, 32,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#           4,  4]], device='cuda:0')
# ```
# 
# Then we will use the id_to_role dictionary to decode to actual tokens.
# 
# ```
# ['[CLS]', 'O', 'O', 'O', 'O', 'O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'O', 'B-V', 'B-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG1', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'O', 'I-ARGM-TMP', 'O', 'I-ARGM-TMP', 'I-ARG2', 'I-ARG1', 'O', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'O', '[SEP]', 'O', 'O', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'O', 'O', 'I-ARGM-TMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG1', 'B-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG2', 'I-ARG1', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'O', 'O', 'O', 'O', 'I-ARG0', 'O', 'O', 'O', 'O', 'O', 'O', 'I-ARG2', 'O', 'O', 'O', 'I-ARG2', 'O', 'I-ARG1', 'O', 'I-ARG2', 'I-ARG1', 'O', 'I-ARG2', 'O', 'O', 'O', 'O', 'O', 'I-ARG2', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# ```
# 
# For now, we just make sure we understand how to do this for a single example. Later, we will write a more formal function to do this once we have trained the model.

# In[17]:


# The outputs are raw logits with shape (batch_size, sequence_length, num_labels)
# Apply argmax to get the predicted label index for each token (batch_size, sequence_length)
predicted_labels = torch.argmax(outputs, dim=-1)

# Flatten the predicted labels for easier access
predicted_labels = predicted_labels.view(-1)  # Flatten to (batch_size * sequence_length)
print(predicted_labels)

# Convert predicted label indices to their corresponding label names using id_to_role
predicted_label_names = [id_to_role.get(label.item(), -100) for label in predicted_labels]

# Convert the input tokens to their string representations
input_tokens = tokenizer.convert_ids_to_tokens(ids.view(-1).cpu().numpy())

# Print the results for comparison
print(f"Input Tokens: {input_tokens}")
print(f"Predicted Labels (indices): {predicted_labels}")
print(f"Predicted Labels (names): {predicted_label_names}")


# ## 3. Training loop

# pytorch provides a DataLoader class that can be wrapped around a Dataset to easily use the dataset for training. The DataLoader allows us to easily adjust the batch size and shuffle the data.

# In[18]:


from torch.utils.data import DataLoader
loader = DataLoader(data, batch_size = 32, shuffle = True)


# The following cell contains the main training loop. The code should work as written and report the loss after each batch,
# cumulative average loss after each 100 batches, and print out the final average loss after the epoch.
# 
# Let us modify the training loop below so that it also computes the accuracy for each batch and reports the
# average accuracy after the epoch.
# The accuracy is the number of correctly predicted token labels out of the number of total predictions.
# We ensure that we exclude [PAD] tokens, i.e. tokens for which the target label is -100. It's okay to include [CLS] and [SEP] in the accuracy calculation.

# In[19]:


loss_function = CrossEntropyLoss(ignore_index = -100, reduction='mean')

LEARNING_RATE = 1e-05
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

device = 'cuda'

def train():
    """
    Train the model for one epoch.
    """
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    correct_predictions = 0
    total_predictions = 0
    # put model in training mode
    model.train()

    for idx, batch in enumerate(loader):

        # Get the encoded data for this batch and push it to the GPU
        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)
        pred_mask = batch['pred'].to(device, dtype = torch.long)

        # Run the forward pass of the model
        logits = model(input_ids=ids, attn_mask=mask, pred_indicator=pred_mask)
        loss = loss_function(logits.transpose(2,1), targets)
        tr_loss += loss.item()
        print("Batch loss: ", loss.item()) # can comment out if too verbose.

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if idx % 100==0:
            #torch.cuda.empty_cache() # can help if you run into memory issues
            curr_avg_loss = tr_loss/nb_tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # Compute accuracy for this batch
        # Exclude [PAD] tokens by ignoring targets == -100
        mask = targets != -100  # mask for valid tokens (non-PAD tokens)

        # Get the predicted labels (argmax over logits)
        predicted_labels = torch.argmax(logits, dim=2)

        # Count correct predictions: compare predicted labels with the true targets
        correct_predictions += torch.sum((predicted_labels == targets) & mask).item()
        total_predictions += torch.sum(mask).item()

        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    avg_accuracy = correct_predictions / total_predictions  # Calculate accuracy for the epoch

    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {avg_accuracy * 100:.2f}%")


# Now let's train the model for one epoch. This will take a while (up to a few hours).

# In[20]:


train()


# In my experiments, I found that two epochs are needed for good performance.

# In[21]:


train()


# I ended up with a training loss of about 0.21 and a training accuracy of 93.65%. Specific values may differ.
# 
# At this point, it's a good idea to save the model (or rather the parameter dictionary) so we can continue evaluating the model without having to retrain.

# In[22]:


torch.save(model.state_dict(), "srl_model_fulltrain_2epoch_finetune_1e-05.pt")


# ## 4. Decoding

# In[28]:


# Optional step: If you stopped working after part 3, first load the trained model

model = SrlModel().to('cuda')
model.load_state_dict(torch.load("srl_model_fulltrain_2epoch_finetune_1e-05.pt"))
model = model.to('cuda')


# Now that we have a trained model, let's try labelling an unseen example sentence. The function decode_output takes the logits returned by the model, extracts the argmax to obtain the label predictions for each token, and then translate the result into a list of string labels. The function label_sentence takes a list of input tokens and a predicate index, prepares the model input, calls the model and then calls decode_output to produce a final result.
# 
# Note that we have already implemented all components necessary (preparing the input data from the token list and predicate index, decoding the model output). But now we are putting it together in one convenient function.

# In[29]:


tokens = "AUN. team spent an hour inside the hospital, where it found evident signs of shelling and gunfire.".split()


# In[30]:


def decode_output(logits): # it will be useful to have this in a separate function later on
    """
    Given the model output, return a list of string labels for each token.
    """
    # Apply argmax to get the predicted label index for each token
    predicted_labels = torch.argmax(logits, dim=2)  # Shape: (batch_size, seq_len)

    # Flatten the tensor to (seq_len,) for easy access
    predicted_labels = predicted_labels.view(-1)

    # Convert the predicted label indices to their corresponding label names using id_to_role
    predicted_label_names = [id_to_role.get(label.item(), -100) for label in predicted_labels]

    return predicted_label_names


# In[31]:


def label_sentence(tokens, pred_idx):

    # complete this function to prepare token_ids, attention mask, predicate mask, then call the model.
    # Decode the output to produce a list of labels.
    # Tokenize the input tokens (turn them into token IDs)
    tokens_with_specials = ['[CLS]'] + tokens + ['[SEP]']

    padding_length = 128 - len(tokens_with_specials)
    if padding_length > 0:
        tokens_with_specials = tokens_with_specials + ['[PAD]'] * padding_length
    else:
        tokens_with_specials = tokens_with_specials[:128]

    token_ids = tokenizer.convert_tokens_to_ids(tokens_with_specials)

    # Create the attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(tokens_with_specials) + [0] * (128 - len(tokens_with_specials))

    # Create the predicate indicator mask (1 at the predicate index, 0 elsewhere)
    predicate_mask = [0] * 128
    predicate_mask[pred_idx + 1] = 1  # Add 1 to pred_idx due to the [CLS] token at the start

    # Convert the inputs to PyTorch tensors and move them to the GPU
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    predicate_mask = torch.tensor(predicate_mask).unsqueeze(0).to(device)

    # Call the model to get the logits
    #vwith torch.no_grad():  # No need to track gradients for inference
    logits = model(input_ids=token_ids, attn_mask=attention_mask, pred_indicator=predicate_mask)

    # Decode the logits into string labels
    label_predictions = decode_output(logits)

    return label_predictions


# In[32]:


# Now you should be able to run

label_test = label_sentence(tokens, 13) # Predicate is "found"
zip(tokens, label_test)

# Print the results
for token, label in zip(tokens, label_test):
    print(f"({token}, {label})")


# The expected output is somethign like this:
# ```   
#  ('A', 'O'),
#  ('U.', 'O'),
#  ('N.', 'O'),
#  ('team', 'O'),
#  ('spent', 'O'),
#  ('an', 'O'),
#  ('hour', 'O'),
#  ('inside', 'O'),
#  ('the', 'B-ARGM-LOC'),
#  ('hospital', 'I-ARGM-LOC'),
#  (',', 'O'),
#  ('where', 'B-ARGM-LOC'),
#  ('it', 'B-ARG0'),
#  ('found', 'B-V'),
#  ('evident', 'B-ARG1'),
#  ('signs', 'I-ARG1'),
#  ('of', 'I-ARG1'),
#  ('shelling', 'I-ARG1'),
#  ('and', 'I-ARG1'),
#  ('gunfire', 'I-ARG1'),
#  ('.', 'O'),
# ```
# 

# ### 5. Evaluation 1: Token-Based Accuracy
# We want to evaluate the model on the dev or test set.

# In[33]:


dev_data = SrlData("propbank_dev.tsv") # Takes a while because we preprocess all data offline


# In[34]:


from torch.utils.data import DataLoader
loader = DataLoader(dev_data, batch_size = 1, shuffle = False)


# In[35]:


# Optional: Load the model again if you stopped working prior to this step.
# model = SrlModel()
# model.load_state_dict(torch.load("srl_model_fulltrain_2epoch_finetune_1e-05.pt"))
# model = mode.to('cuda')


# Let us complete the evaluate_token_accuracy function below. The function should iterate through the items in the data loader (see training loop in part 3). Running the model on each sentence/predicate pair and extract the predictions.
# 
# For each sentence, counting the correct predictions and the total predictions. Finally, computing the accuracy as #correct_predictions / #total_predictions
# 
# Careful: We need to filter out the padded positions ([PAD] target tokens), as well as [CLS] and [SEP]. It's okay to include [B-V] in the count though.

# In[43]:


def evaluate_token_accuracy(model, loader):

    model.eval() # put model in evaluation mode

    # Variables to track accuracy
    total_correct = 0  # Number of correct predictions
    total_predictions = 0  # Total number of tokens considered for accuracy

    # Iterate over the data loader (this will return batches)
    for batch in loader:
        # Assuming `batch` is a tuple: (input_ids, attention_mask, predicate_mask, labels)
        input_ids, attention_mask, predicate_mask, labels = batch

        # Move tensors to the correct device (e.g., GPU if available)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        predicate_mask = predicate_mask.to(model.device)
        labels = labels.to(model.device)

        # Run the model on the input data
        with torch.no_grad():  # Disable gradient calculation during evaluation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, predicate_mask=predicate_mask)

        # Get the predicted token labels (argmax of logits)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Flatten the predictions and labels to compare them token by token
        predictions = predictions.flatten()
        labels = labels.flatten()

        # Iterate over each token in the batch (ignoring padding tokens)
        for pred, label in zip(predictions, labels):
            # We will ignore [PAD], [CLS], and [SEP] tokens
            if label.item() == role_to_id['[PAD]']:
                continue  # Skip padding tokens
            if label.item() == role_to_id['[CLS]'] or label.item() == role_to_id['[SEP]']:
                continue  # Skip [CLS] and [SEP] tokens

            # Count correct predictions
            total_predictions += 1
            if pred.item() == label.item():
                total_correct += 1

    # Compute accuracy
    if total_predictions > 0:
        acc = total_correct / total_predictions
        print(f"Accuracy: {acc}")
    else:
        print("No valid tokens to evaluate.")


# ### 6. Span-Based evaluation
# 
# While the accuracy score in part 5 is encouraging, an accuracy-based evaluation is problematic for two reasons. First, most of the target labels are actually 0. Second, it only tells us that per-token prediction works, but does not directly evaluate the SRL performance.
# 
# Instead, SRL systems are typically evaluated on micro-averaged precision, recall, and F1-score for predicting labeled spans.
# 
# More specifically, for each sentence/predicate input, we run the model, decode the output, and extract a set of labeled spans (from the output and the target labels). These spans are (i,j,label) tuples.  
# 
# We then compute the true_positives, false_positives, and false_negatives based on these spans.
# 
# In the end, we can compute
# 
# * Precision:  true_positive / (true_positives + false_positives)  , that is the number of correct spans out of all predicted spans.
# 
# * Recall: true_positives / (true_positives + false_negatives) , that is the number of correct spans out of all target spans.
# 
# * F1-score:   (2 * precision * recall) / (precision + recall)
# 
# 
# For example, consider
# 
# | |[CLS]|The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
# |--||---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|             
# ||0|1|2|3|4|5|6|7|8|9|1O|11|12|13|14|15|
# |target|[CLS]|B-ARG1|I-ARG1|B-V|B-ARG2|I-ARG2|I-ARG2|I-ARG2|I-ARG2|O|O|O|O|O|O|O|
# |prediction|[CLS]|B-ARG1|I-ARG1|B-V|I-ARG2|I-ARG2|O|O|O|O|O|O|O|O|B-ARGM-TMP|O|
# 
# The target spans are (1,2,"ARG1"), and (4,8,"ARG2").
# 
# The predicted spans would be (1,2,"ARG1"), (14,14,"ARGM-TMP"). Note that in the prediction, there is no proper ARG2 span because we are missing the B-ARG2 token, so this span should not be created.
# 
# So for this sentence we would get: true_positives: 1 false_positives: 1 false_negatives: 1
# 
# Let us complete the function evaluate_spans that performs the span-based evaluation on the given model and data loader. We can use the provided extract_spans function, which returns the spans as a dictionary. For example
# {(1,2): "ARG1", (4,8):"ARG2"}

# In[44]:


def extract_spans(labels):
    spans = {} # map (start,end) ids to label
    current_span_start = 0
    current_span_type = ""
    inside = False
    for i, label in enumerate(labels):
        if label.startswith("B"):
            if inside:
                if current_span_type != "V":
                    spans[(current_span_start,i)] = current_span_type
            current_span_start = i
            current_span_type = label[2:]
            inside = True
        elif inside and label.startswith("O"):
            if current_span_type != "V":
                spans[(current_span_start,i)] = current_span_type
            inside = False
        elif inside and label.startswith("I") and label[2:] != current_span_type:
            if current_span_type != "V":
                spans[(current_span_start,i)] = current_span_type
            inside = False
    return spans


# In[45]:


def evaluate_spans(model, loader):


    total_tp = 0
    total_fp = 0
    total_fn = 0

    model.eval()

    for idx, batch in enumerate(loader):

        # Assuming batch is a tuple: (input_ids, attention_mask, predicate_mask, labels)
        input_ids, attention_mask, predicate_mask, labels = batch

        # Move tensors to the correct device
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        predicate_mask = predicate_mask.to(model.device)
        labels = labels.to(model.device)

        # Get the model predictions (logits)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, predicate_mask=predicate_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)  # Get the index of the highest probability (predicted label)

        # Flatten the predictions and labels for easier comparison
        predictions = predictions.flatten()
        labels = labels.flatten()

        # Extract spans from the predictions and labels
        predicted_spans = extract_spans([id_to_role[label.item()] for label in predictions])
        target_spans = extract_spans([id_to_role[label.item()] for label in labels])

        # Calculate true positives, false positives, and false negatives
        for span in predicted_spans:
            if span in target_spans and target_spans[span] == predicted_spans[span]:
                total_tp += 1
            else:
                total_fp += 1

        for span in target_spans:
            if span not in predicted_spans:
                total_fn += 1

    # Calculate Precision, Recall, and F1-score
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f = (2 * total_p * total_r) / (total_p + total_r) if (total_p + total_r) > 0 else 0

    print(f"Overall P: {total_p}  Overall R: {total_r}  Overall F1: {total_f}")

#evaluate(model, loader)


# In my evaluation, I got an F score of 0.82  (which slightly below the state-of-the art in 2018)
