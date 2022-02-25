import torch
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
from transformers import BertModel
from torch import nn


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label-1 for label in df['Class Index']]
        self.texts = [tokenizer(title+description,
                                padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for title,description in zip(df['Title'],df['Description'])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
Model_path = '/raid/okano/model/txt_cls_bert/2022_02_24_08_44_59'
select_num = 3
model_name = f'/cls_t_prms_{select_num}.pth'

model = BertClassifier()

test_datapath = 'data/test.csv'
df_test = pd.read_csv(test_datapath)

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.load_state_dict(torch.load(Model_path+model_name, map_location=device))
if use_cuda:
    model = model.cuda()
test = False
if test:
    evaluate(model, df_test)

def predict(inputtext):
    predicted_labels = []
    with torch.no_grad():
        inputtext_dataset = input_to_dataset(inputtext)
        inputtext_dataloader = torch.utils.data.DataLoader(inputtext_dataset)
        print(f'len(test) {len(inputtext_dataset)}')
        print(f'len(test_dataloader) {len(inputtext_dataloader)}')
        for input_info__, _ in tqdm(inputtext_dataloader):
            mask = input_info__['attention_mask'].to(device)
            input_id = input_info__['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            print(output)
            predicted_labels.append(output.argmax(dim = 1) + 1)

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

Classifyinputtext = True
input = [
"In the span of a week, three American-born athletes of Chinese descent have been thrust into the spotlight at the Beijing Winter Olympics -- to very different reactions in China.\
All three were trained in the United States and are only a few years apart in age, but their paths diverged on the way to the Games: freestyle skier Eileen Gu and figure skater Zhu Yi chose to compete for China, while Nathan Chen, another figure skater, opted for Team USA.\
Gu and Chen both won gold, while Zhu faltered on the ice during two consecutive showings. \
The public responses they've received in the Olympic host nation also took different turns."
,
"At least eight Westerners have been arrested by the Taliban in Afghanistan during different incidents in the last two months, CNN has learned, marking a sharp escalation of Taliban actions against Westerners living in the country.\
No formal charges appear to have been lodged against the detained men. \
They include seven British citizens including one who is an American legal resident and one US citizen, according to the sources with direct knowledge of the matter in Afghanistan, the United States, and the UK.\
The former vice president of Afghanistan, Amrullah Saleh tweeted that \"nine\" Westerners had been \"kidnapped\" by the Taliban, naming journalists Andrew North, formerly of the BBC who was in the country working for the United Nations and Peter Jouvenal, who has worked with the BBC and CNN, both are British citizens.\
The reason for each specific detention is unclear, and they are not thought all to be related.\
Jouvenal's detention was confirmed by his family and friends to CNN."
,
"Imagine having filed and paid your taxes last year, then months later you get a letter in the mail from the IRS saying you didn't.\
That's what's happening to many taxpayers this year thanks to automated notices being sent by the IRS.\
But if you got one, don't panic. \
There's a fair chance the IRS simply hasn't seen what you already sent in. \
That's because it's dealing with a mountain of returns and correspondence that has built up over the past two years. \
During that time, the agency was called on to deliver several rounds of economic impact payments and other financial Covid-19 relief, while trying to protect its own workforce from Covid."
,
"Late Monday night, some Peloton (PTON) staffers noticed they were unable to access work productivity apps like Slack and Okta, which they used regularly on the job. \
Peloton's employees had been told about a scheduled maintenance window that might cause service outages, according to one employee, but that didn't stop others from bracing for the worst.\
\"I'm freaking out,\" another former Peloton employee who worked in the company's product department recalled to CNN Business. \
He said coworkers frantically texted each other as the\
y speculated about what the morning might bring. \
Peloton was reporting its earnings Tuesday, and weeks earlier the CEO said the company was reviewing its costs and that layoffs were on the table."
,
"AP - Southern California's smog-fighting agency went after emissions of the bovine variety Friday, adopting the nation's first rules to reduce air pollution from dairy cow manure."
]

class input_to_dataset(torch.utils.data.Dataset):

    def __init__(self, input):

        self.labels = [-1 for _ in input]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in input]
        print(self.labels)
        print(self.texts)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

if Classifyinputtext:
    predicted_labels = predict(input)
    for text, predicted_label in zip(input,predicted_labels):
        print(text)
        print(f'This is a {ag_news_label[int(predicted_label[0]+1)]} news.')