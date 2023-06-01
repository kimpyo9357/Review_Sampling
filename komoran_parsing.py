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

if __name__ == "__main__":
    file_name = ['reveiew_13276568.txt']
    for file in file_name:
        komoran_parsing(file)