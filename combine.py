#-*- coding: utf-8 -*-
import re
from PyKomoran import *

def komoran_parsing(file):
    komoran = Komoran("EXP")
    #file_name = 'reveiew_13276568.txt'
    file_name = file
    fr = open(file_name,'r',encoding='UTF8')
    pr_fw = open('pre_'+file_name,'w',encoding='UTF8')
    fw = open(file_name[:-4]+'_komoran'+file_name[-4:],'w',encoding='UTF8')

    ret_komoran = []
    ret_text = []

    while True :
        line = fr.readline()
        if not line : break
        line = re.sub(r'[^가-힣a-zA-Z0-9\s\.]','',line)
        line = line.replace('\n', '')
        line = list(line)

        for i in range(len(line)-1,0,-1):
            if i-1 > 0:
                if line[i] == '.' and line[i-1] == line[i]:
                    line.pop(i)
                elif line[i] == ' ' and line[i-1] == line[i]:
                    line.pop(i)

        line2 = line[:]
        line2 = ''.join(line2)
        line2 = line2.replace('.','.\n')
        line2 = line2.split(' ')
        line = ''.join(line).split(' ')

        pre_line = []
        index = [0]
        check = 0
        for i in range(len(line)):
            temp = komoran.get_plain_text(line[i])
            temp = temp.replace('./SP', './SF')
            count = 0
            for j in temp.split():
                token = j.split('/')
                count += 1
                if (len(token) != 2 or token[1] == 'NA'):
                    check = 1
                pre_line.append(j)
            index.append(count)
        if check == 1:
            continue
        index_len = len(pre_line)

        i = 0
        count = 0

        if (pre_line[-1] !='/SW'):
            pre_line.append('/SW')
            index_len += 1
        while (i < index_len):
            temp = pre_line[i].split('/')
            try:
                if (i == index[count+1]):
                    count += 1
                    index[count+1] = index[count+1] + index[count]
                if ((temp[0][-1] =='요' and (temp[1] == 'EC' or temp[1] == 'EF') or (temp[0][-1] =='다' and temp[1] == 'EF')) and pre_line[i+1].split('/')[1] != 'SF'):
                    pre_line.insert(i+1,'./SF')
                    pre_line[i] = pre_line[i].replace('EC','EF')
                    if temp[0][-1] =='요':
                        line2[count] = line2[count][:line2[count].find('요')+1] + '.\n' + line2[count][line2[count].find('요')+1:]
                    elif temp[0][-1] =='다':
                        line2[count] = line2[count][:line2[count].find('다')+1] + '.\n' + line2[count][line2[count].find('다')+1:]
                    else:
                        line2[count] = line2[count].replace(' ', '')
                        line2[count] += '.\n'
                    index_len += 1
                    index[count+1] += 1
                if (temp[1][0]=='S' and pre_line[i+1].split('/')[1] == 'SF'):
                    pre_line[i+1] = pre_line[i+1].replace('SF','SP')
                    line2[count] = line2[count].replace(temp[0]+".\n",temp[0]+".")
                if ('.' in temp[0]) and temp[1][0] == 'N':
                    line2[count] = line2[count].replace('.\n','.')
                if (pre_line[i+1].split('/')[1] == 'SW' and temp[1] !='SF'):
                    pre_line.insert(i+1,'./SF')
                    pre_line[i] = pre_line[i].replace('EC', 'EF')
                    line2[count] = line2[count].replace(' ', '')
                    line2[count] += '.\n'
                    index_len += 1
                    index[count + 1] += 1
            except:
                pass
            i += 1

        pre_line = ' '.join(pre_line)
        pre_line = pre_line.replace('/SF ', '/SF\n')
        pre_line = pre_line.replace('/SW', '/SW\n')
        line2 = ' '.join(line2)
        line2 = line2.replace('\n ','\n')
        line2 += '<EOL>\n'

        pr_fw.write(line2)
        fw.write(pre_line)

        ret_text.append(line2)
        ret_komoran.append(pre_line)

    ret_komoran = ''.join(ret_komoran).split('\n')
    ret_text = ''.join(ret_text).split('\n')
    ret_komoran.pop(-1)
    ret_text.pop(-1)
    return ret_komoran, ret_text

def textrank_file(file):
    file_name = file
    with open(file_name[:-4] + '_komoran' + file_name[-4:], encoding='utf-8') as f:
        sents = [sent.strip() for sent in f]

    with open('pre_' + file_name, encoding='utf-8') as f:
        texts = [sent.strip() for sent in f]

    from textrank import KeywordSummarizer

    def komoran_tokenize(sent):
        words = sent.split()
        words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    keyword_extractor = KeywordSummarizer(
        tokenize=komoran_tokenize,
        window=-1,
        verbose=False
    )
    keywords = keyword_extractor.summarize(sents, topk=30)

    frequent_words = set()
    for word, rank in keywords:
        if word[-3:] in {'NNG', 'NNP'}:
            if len(frequent_words) < 10:
                frequent_words.add(word[:-4])

    frequent_words = list(frequent_words)
    frequent_words.sort()

    fm = open(file_name[:-4] + '_main_key' + file_name[-4:], 'w', encoding='UTF8')
    fm.write(' '.join(frequent_words) + '\n')

    fs = open(file_name[:-4] + '_split_text_sub' + file_name[-4:], 'w', encoding='UTF8')
    fsk = open(file_name[:-4] + '_split_text_sub_komoran' + file_name[-4:], 'w', encoding='UTF8')

    for i in range(len(texts)):
        index = 0
        sub_list = [0 for i in range(10)]
        for word in range(len(frequent_words)):
            if frequent_words[word] in texts[i]:
                sub_list[index] = 1
                s_num = 1
            index += 1
        if s_num == 1:
            fm.write(' '.join(map(str, sub_list)) + '\n')
            fs.write(texts[i] + '\n')
            fsk.write(sents[i] + '\n')
        s_num = 0
    f.close()
    fm.close()

def textrank_list(komoran,text):
    sents = komoran
    texts = text

    from textrank import KeywordSummarizer

    def komoran_tokenize(sent):
        words = sent.split()
        words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    keyword_extractor = KeywordSummarizer(
        tokenize=komoran_tokenize,
        window=-1,
        verbose=False
    )
    keywords = keyword_extractor.summarize(sents, topk=30)

    frequent_words = set()
    for word, rank in keywords:
        if word[-3:] in {'NNG', 'NNP'}:
            if len(frequent_words) < 10 and len(word[:-4]) != 1 and word[:-4] != "배송":
                frequent_words.add(word[:-4])

    frequent_words = list(frequent_words)
    frequent_words.sort()

    key = []
    key.append(frequent_words)
    ret_text = []
    ret_komoran = []

    s_num = 0
    for i in range(len(texts)):
        index = 0
        sub_list = [0 for i in range(10)]
        for word in range(len(frequent_words)):
            if frequent_words[word] in texts[i]:
                sub_list[index] = 1
                s_num = 1
            index += 1
        if s_num == 1:
            key.append(' '.join(map(str, sub_list)))
            ret_text.append(texts[i])
            ret_komoran.append(sents[i])
        s_num = 0

    return key, ret_komoran, ret_text

import bert


if __name__ == "__main__":
    file_name = ['reveiew_13276568.txt']
    for file in file_name:
        komoran, pre_text = komoran_parsing(file)
        key, komoran, pre_text = textrank_list(komoran,pre_text)
        pros_cons = bert.BERT_list(key,komoran)
    print(pros_cons)