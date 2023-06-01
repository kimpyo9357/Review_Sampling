
file_name = 'reveiew_13276568.txt'
with open(file_name[:-4]+'_komoran'+file_name[-4:], encoding='utf-8') as f:
    sents = [sent.strip() for sent in f]

with open('pre_'+file_name, encoding='utf-8') as f:
    texts = [sent.strip() for sent in f]

print(sents)
print(texts)

from textrank import KeywordSummarizer

def komoran_tokenize(sent):
    words = sent.split()
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

keyword_extractor = KeywordSummarizer(
    tokenize = komoran_tokenize,
    window = -1,
    verbose = False
)
keywords = keyword_extractor.summarize(sents, topk=30)

frequent_words = set()
for word, rank in keywords:
    if word[-3:] in {'NNG','NNP'}:
        if len(frequent_words) < 10:
            frequent_words.add(word[:-4])

frequent_words = list(frequent_words)
frequent_words.sort()

fm = open(file_name[:-4]+'_main_key'+file_name[-4:],'w',encoding='UTF8')
fm.write(' '.join(frequent_words)+'\n')

fs = open(file_name[:-4]+'_split_text_sub'+file_name[-4:],'w',encoding='UTF8')
fsk = open(file_name[:-4]+'_split_text_sub_komoran'+file_name[-4:],'w',encoding='UTF8')
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
        fm.write(' '.join(map(str,sub_list)) + '\n')
        fs.write(texts[i]+'\n')
        fsk.write(sents[i]+'\n')
    s_num = 0
f.close()
fm.close()