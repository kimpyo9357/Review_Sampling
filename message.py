import pika, sys, os,json, threading
import time
import crawling
import combine
import bert
import mysql_command

connection = pika.BlockingConnection(pika.ConnectionParameters(host='14.47.104.94'))
channel = connection.channel()

child_thread = True
db = mysql_command.mysql

def main():
    global child_thread
    print(' [*] Waiting for messages. To exit press CTRL+C')

    channel.queue_declare(queue='spring.python.product') # spring -> python
    channel.queue_declare(queue='python.spring.product')
    channel.basic_consume(queue='spring.python.product', on_message_callback=read_instruction,auto_ack=True)
    '''if child_thread == 1:
        t = threading.Thread(target=read_instruction) ## 스레드 실행 확인 필요
        t.start()
        child_thread -= 1
    time.sleep(5)'''
    channel.start_consuming()

'''def callback(ch, method, properties, body):
    global child_thread
    child_thread = True
    with open('./data.json','r', encoding='UTF8') as file:
        data = file.read()
    channel.basic_publish(exchange='', routing_key='python.spring.product', body=data)
    print(" [x] Sent 'Hello World!'")
    print("callback")
    print(body)'''

def read_instruction(ch, method, properties, body):
    global db
    inst_data = body.split(',')
    try:
        if (inst_data[0] == 'search'): ## list crawling
            datas = crawling.list_crawling(inst_data[1])  ## 파일 리스트 출력 -> data.json
            with open("data.json", 'w', encoding='utf-8') as f:
                json.dump(datas, f, ensure_ascii=False, indent='\t')
            channel.basic_publish(exchange='', routing_key='python.spring.product', body=datas)
            for i in datas.keys():
                data = [i, datas[i]['name'], datas[i]['category']]
                detail_data = [i,datas[i]['name'],datas[i]['detail']]
                db.insert('product',data)
                db.insert('productdetail',detail_data)
        elif (inst_data[0] == 'review'): ## review crawling
            review = dict()
            pcode = crawling.ret_pcode(inst_data[1])
            db_productanal = db.search('productanal',pcode)
            if len(db_productanal) == 0:
                data = []
                review['pcode'] = pcode
                data.append(review['pcode'])
                db_product = db.search('product', pcode)
                review['name'] = db_product[1]
                data.append(review['name'])
                crawling.find_review(pcode)
                print("pass crawling " + str(pcode))
                komoran, pre_text = combine.komoran_parsing('reveiew_' + str(pcode) + '.txt')
                print("pass morphological analysis " + str(pcode))
                key, komoran, pre_text = combine.textrank_list(komoran, pre_text)
                print("pass key analysis " + str(pcode))
                pros_cons = bert.BERT_list(key, komoran)
                review['pros'] = tuple(list(pros_cons[-3:])[::-1])
                review['cons'] = pros_cons[:3]
                data = data + list(review['pros']) + list(review['cons'])
                db.insert('productanal',data)
            else:
                review['pcode'] = db_productanal[0]
                review['name'] = db_productanal[1]
                review['pros'] = db_productanal[2:5]
                review['cons'] = db_productanal[5:]
            with open('./pros_cons.json', 'w', encoding='UTF-8') as make_file:
                json.dump(review, make_file, indent="\t",ensure_ascii=False)
            channel.basic_publish(exchange='', routing_key='python.spring.product', body=review)  # review 반환
    except:
        print(' [*] Doesn\'t have instruction or not match type')
        #channel.basic_publish(exchange='', routing_key='python.spring.product', body="except") # 명령 파일 없음
    return

if __name__ == '__main__':
    main()