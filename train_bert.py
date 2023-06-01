import pandas as pd
import urllib.request
from wordcloud import STOPWORDS

#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

comment_words = ''
stopwords = set(STOPWORDS)
tokenized_text = []

'''# 3. 문장 데이터를 단어화하기
for val in train_data["document"]:

  # 문장을 string으로 만들기
  val = str(val)

  # 문장을 쪼개기
  tokens = val.split()

  tokenized_text.append(tokens)
  comment_words += " ".join(tokens) + " "


stopwords_2 = ["영화", "진짜", "정말", "이거", "그냥", "너무", "영화가", "영화는",
             "이거", "이게", "이건", "영화의", "어떤", "아주", "계속", "영화다",
             "영화를", "그리고"
]


def filter_stopwords(tokenized_text, stopwords_2):
  tokenized_filtered = []

  for i in tokenized_text:
    for word in i:
      if word not in stopwords and word not in stopwords_2:
        tokenized_filtered.append(word)

  return tokenized_filtered

import operator


def word_count(tokenized_data):
  word_counter = {}

  for i in tokenized_data:
    if i in word_counter.keys():
      word_counter[i] += 1
    else:
      word_counter[i] = 1

  # 많이 나온 순서대로 정렬

  sorted_dict = dict(sorted(word_counter.items(),
                            key=operator.itemgetter(1), reverse=True))

  return sorted_dict

def top_20(tokenized_dict):
  top_20_words = list(tokenized_dict.items())[:20]
  return top_20_words

stopwords_2.extend(["이", "이렇게", "더", "수", "다", "그", "내가", "이렇게",
               "완전", "봤는데", "영화.", "평점", "평점이", "왜", "이런", "본",
               "보고", "잘"
])

stopwords_2.extend(["보는", "내", "다시", "난", "연기", "한", "것", "하는", "또",
                    "역시", "좀", "참", "많이", "없는", "있는"
])

tokenized_filtered = filter_stopwords(tokenized_text, stopwords_2)
tokenized_dict = word_count(tokenized_filtered)

#print(top_20(tokenized_dict))

emotion_dict = {"최고의": "극찬", "ㅋㅋ": "웃음", "좋은": "기쁨", "재밌게": "흥미",
                "쓰레기": "혐오", "ㅋㅋㅋ": "웃음", "ㅋ": "무심", "ㅠㅠ": "슬픔"
}

from collections import defaultdict, OrderedDict

emotions_dict = defaultdict(int)

emotions_list = []

for k, v in tokenized_dict.items():
  for key, value in emotion_dict.items():
    if k == key:
      emotions_list.append((value, v))

for k, v in emotions_list:
  if k in emotions_dict:
    emotions_dict[k] += v
  else:
    emotions_dict[k] = v

emotions_dict = OrderedDict(sorted(emotions_dict.items(),
                            key=lambda item: item[1],
                            reverse=True))
'''
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, Adafactor, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import os
import datetime


from PyKomoran import *
komoran = Komoran("EXP")

os.makedirs(f'./saved_models', exist_ok=True)

##train 토큰화
train_data_values = train_data['label'].values[:100]
test_data_values = test_data["label"].values[:100]

tokenized_data = []
count = 0
#for i in range(len(train_data['document'])):
for i in range(0,100):
  try:
    print(str(i) +"/"+str(len(train_data['document'])))
    temp = komoran.get_plain_text(train_data['document'][i])
    bert = temp.split()
    for j in range(len(bert)):
      bert[j] = bert[j].split("/")[0]
    bert.insert(0,"[CLS]")
    bert.append("[SEP]")
    tokenized_data.append(bert)
  except:
    train_data_values = np.delete(train_data_values, i-count)
    count += 1
    print("pass")
    continue

##test 토큰화
tokenized_test_data = []
count = 0
for i in range(0,100):
  try:
    print(str(i) +"/"+str(len(test_data['document'])))
    temp = komoran.get_plain_text(test_data['document'][i])
    bert = temp.split()
    for j in range(len(bert)):
      bert[j] = bert[j].split("/")[0]
    bert.insert(0,"[CLS]")
    bert.append("[SEP]")
    tokenized_test_data.append(bert)
  except:
    test_data_values = np.delete(test_data_values, i-count)
    count += 1
    print("pass")
    continue

#print(test_data_values)
#전처리
################################################################################################################
#BERT 학습
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
# padding
#input_ids = []
'''for i in tokenized_data:
  ids = tokenizer.convert_tokens_to_ids(i)
  input_ids.append(ids)'''

#print(input_ids)

input_ids_test = []
for i in tokenized_test_data:
  ids = tokenizer.convert_tokens_to_ids(i)
  input_ids_test.append(ids)

max_len = 128
#input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')
input_ids_test = pad_sequences(input_ids_test, maxlen=max_len, dtype='long', truncating='post', padding='post')

attention_masks = []

'''for ids in input_ids:
  ids_mask = []
  for id in ids:
    masked = float(id > 0)
    ids_mask.append(masked)
  attention_masks.append(ids_mask)'''

attention_masks_test = []

for ids in input_ids_test:
  ids_mask = []
  for id in ids:
      masked = float(id>0)
      ids_mask.append(masked)
  attention_masks_test.append(ids_mask)


'''X_train, X_val, y_train, y_val = train_test_split(
    input_ids, train_data_values, random_state=42, test_size=0.2)'''

'''print(X_train)
print("------------------------------------------------")
print(X_val)
print("------------------------------------------------")
print(y_train)
print("------------------------------------------------")
print(y_val)
print("------------------------------------------------")'''


#train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.2)

'''print(train_masks)
print("------------------------------------------------")
print(validation_masks)
print("------------------------------------------------")

X_train_tune = X_train
y_train_tune = y_train
X_val_tune = X_val
y_val_tune = y_val'''

X_test_tensor = input_ids_test
y_test_tensor = test_data_values
test_masks = attention_masks_test

'''# PyTorch로 변환
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
train_masks = torch.tensor(train_masks)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)
validation_masks = torch.tensor(validation_masks)'''

X_test_tensor = torch.tensor(X_test_tensor)
y_test_tensor = torch.tensor(y_test_tensor)
test_masks = torch.tensor(attention_masks_test)

batch_size = 32
'''
train = TensorDataset(X_train_tensor, train_masks, y_train_tensor)
train_sampler = RandomSampler(train)

val = TensorDataset(X_val_tensor, validation_masks, y_val_tensor)
val_sampler = SequentialSampler(val)
'''

test = TensorDataset(X_test_tensor, test_masks, y_test_tensor)
test_sampler = RandomSampler(test)

'''train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size)
val_dataloader = DataLoader(val, sampler=val_sampler, batch_size=batch_size)'''
test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()

for param in model.base_model.parameters():
    param.requires_grad = False

from sklearn.model_selection import train_test_split

train_queries, val_queries, train_docs, val_docs, train_labels, val_labels = train_test_split(
  train_data["id"].apply(str).tolist(),
  train_data["document"].apply(str).tolist(),
  train_data["label"].tolist(),
  test_size=.2
)

from transformers import BertTokenizerFast

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_queries, train_docs, truncation=True, padding='max_length', max_length=max_len)
val_encodings = tokenizer(val_queries, val_docs, truncation=True, padding='max_length', max_length=max_len)

import torch

class Cord19Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
'''
train_dataset = Cord19Dataset(train_encodings, train_labels)
val_dataset = Cord19Dataset(val_encodings, val_labels)

from transformers import Trainer, TrainingArguments

t raining_args = TrainingArguments(
    output_dir='./',          # output directory
    evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=0,               # strength of weight decay
    save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
'''
# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # 학습률
                  eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값,
                  )
# 에폭수
epochs = 30

# 총 훈련 스텝
#total_steps = len(train_dataloader) * epochs

# Learning rate decay를 위한 스케줄러
'''scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)'''

# 정확도 계산 함수
def accuracy_measure(y_pred, y):
    pred_flattened = np.argmax(y_pred, axis=1).flatten()
    print(pred_flattened)
    y_flattened = y.flatten()
    return np.sum(pred_flattened == y_flattened) / len(y_flattened)

# 시간 표시 함수
def time_elapsed(elapsed):
    # 반올림
    elapsed = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed))

# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

'''
best_accuracy = 0
# 본격적인 학습
for epoch_i in range(0, epochs):

    # ========================================
    #                            Training
    # ========================================

    # 현재 훈련 조건 표시
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = time_elapsed(time.time() - t0)
            print(' Batch {:>5,}    of  {:>5,}.     Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # Forward 수행
        outputs = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)

        # 로스 구함
        loss = outputs[0]
        # 총 로스 계산
        total_loss += loss.item()
        # Backward 수행으로 그래디언트 계산
        loss.backward()
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()
        # 스케줄러로 학습률 감소
        scheduler.step()
        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print(" Average training loss: {0:.2f}".format(avg_train_loss))
    print(" Training epcoh took: {:}".format(time_elapsed(time.time() - t0)))
    # ========================================
    #                            Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in val_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask)

            # 로스 구함
            logits = outputs[0]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = accuracy_measure(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            torch.save(model.state_dict(), f'./saved_models/best_accuracy.pth')
    torch.save(model.state_dict(), f'./saved_models/iter_{epoch_i +1}.pth')
    print(" Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print(" Best_Accuracy: {0:.2f}".format(best_accuracy / nb_eval_steps))
    print(" Validation took: {:}".format(time_elapsed(time.time() - t0)))

print("")
print("Training complete!")
'''
# 시작 시간 설정
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = time_elapsed(time.time() - t0)
        print(' Batch {:>5,}    of  {:>5,}.     Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)

    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = accuracy_measure(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1


print("")
print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(time_elapsed(time.time() - t0)))