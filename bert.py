import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.utils import pad_sequences
from transformers import logging
logging.set_verbosity_error()

import sys
import time
import datetime
import numpy as np

##################### GPU 확인
'''import os

n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))
'''

# 시간 표시 함수
def time_elapsed(elapsed):
    # 반올림
    elapsed = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed))

def BERT_file(file):
    bert_text = []
    file_name = 'reveiew_13276568.txt'
    file_name = file
    with open(file_name[:-4]+'_split_text_sub_komoran'+file_name[-4:],'r',encoding='UTF8') as f:
        sub_texts = [sent.strip() for sent in f]

    with open(file_name[:-4]+'_main_key'+file_name[-4:],'r',encoding='UTF8') as f:
        keys = [sent.strip() for sent in f]

    key = keys[0].split()
    keys.pop(0)
    for i in range(len(keys)):
        keys[i] = list(map(int,keys[i].split()))

    keys = np.array(keys)
    count = keys.sum(axis=0)

    ### 긴 문장으로 했을 경우 다 부정 처리 남. -> 제외
    train_data = sub_texts

    for i in train_data:
      temp = []
      i = i.split()
      for j in i:
        if j.split("/")[1] == "SF":
          temp.insert(0,"[CLS]")
          temp.append("[SEP]")
          bert_text.append(temp)
          temp = []
        else:
            temp.append(j.split("/")[0])

    #BERT 예측
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    # padding
    input_ids = []
    for i in bert_text:
      ids = tokenizer.convert_tokens_to_ids(i)
      input_ids.append(ids)

    max_len = 128
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

    attention_masks = []
    for ids in input_ids:
      ids_mask = []
      for id in ids:
        masked = float(id > 0)
        ids_mask.append(masked)
      attention_masks.append(ids_mask)


    X_test_tensor = input_ids
    test_masks = attention_masks

    X_test_tensor = torch.tensor(X_test_tensor)
    test_masks = torch.tensor(attention_masks)

    batch_size = 32

    test = TensorDataset(X_test_tensor, test_masks)
    test_sampler = RandomSampler(test)

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
    model.load_state_dict(torch.load("./saved_models/best_accuracy.pth", map_location=device))

    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    t0 = time.time()

    # 평가모드로 변경
    model.eval()


    pred_flattened = np.empty(0)
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = time_elapsed(time.time() - t0)
            print(' Batch {:>5,}    of  {:>5,}.     Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids, token_type_ids=None,)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()

        pred_flattened = np.concatenate((pred_flattened,np.argmax(logits, axis=1).flatten()))

    for i in range(len(bert_text)):
      keys[i] = np.multiply(keys[i],pred_flattened[i])

    np.set_printoptions(threshold=sys.maxsize)
    total = keys.sum(axis=0)
    avg = total/count

    pros_cons = {name : value for name, value in zip(key,avg)}
    pros_cons = sorted(pros_cons.items(), key = lambda item: item[1])
    return pros_cons

def BERT_list(keys,komoran):
    bert_text = []
    '''file_name = 'reveiew_13276568.txt'
    file_name = file
    with open(file_name[:-4]+'_split_text_sub_komoran'+file_name[-4:],'r',encoding='UTF8') as f:
        sub_texts = [sent.strip() for sent in f]

    with open(file_name[:-4]+'_main_key'+file_name[-4:],'r',encoding='UTF8') as f:
        keys = [sent.strip() for sent in f]'''

    key = keys
    sub_texts = komoran

    key = keys[0]
    keys.pop(0)
    for i in range(len(keys)):
        keys[i] = list(map(int,keys[i].split()))

    keys = np.array(keys)
    count = keys.sum(axis=0)

    ### 긴 문장으로 했을 경우 다 부정 처리 남. -> 제외
    train_data = sub_texts

    for i in train_data:
      temp = []
      i = i.split()
      for j in i:
        if j.split("/")[1] == "SF":
          temp.insert(0,"[CLS]")
          temp.append("[SEP]")
          bert_text.append(temp)
          temp = []
        else:
            temp.append(j.split("/")[0])

    #BERT 예측
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    # padding
    input_ids = []
    for i in bert_text:
      ids = tokenizer.convert_tokens_to_ids(i)
      input_ids.append(ids)

    max_len = 128
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

    attention_masks = []
    for ids in input_ids:
      ids_mask = []
      for id in ids:
        masked = float(id > 0)
        ids_mask.append(masked)
      attention_masks.append(ids_mask)


    X_test_tensor = input_ids
    test_masks = attention_masks

    X_test_tensor = torch.tensor(X_test_tensor)
    test_masks = torch.tensor(attention_masks)

    batch_size = 32

    test = TensorDataset(X_test_tensor, test_masks)
    test_sampler = RandomSampler(test)

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
    model.load_state_dict(torch.load("./saved_models/best_accuracy.pth", map_location=device))

    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    t0 = time.time()

    # 평가모드로 변경
    model.eval()


    pred_flattened = np.empty(0)
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = time_elapsed(time.time() - t0)
            print(' Batch {:>5,}    of  {:>5,}.     Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids, token_type_ids=None,)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()

        pred_flattened = np.concatenate((pred_flattened,np.argmax(logits, axis=1).flatten()))

    for i in range(len(bert_text)):
      keys[i] = np.multiply(keys[i],pred_flattened[i])

    np.set_printoptions(threshold=sys.maxsize)
    total = keys.sum(axis=0)
    avg = total/count
    print(key)
    print(total)
    print(count)
    print(avg)

    pros_cons = {name : value for name, value in zip(key,avg)}
    pros_cons = sorted(pros_cons.items(), key = lambda item: item[1])
    return pros_cons

if __name__ == "__main__":
    pros_cons = BERT_file('reveiew_13276568.txt')
    print(pros_cons)