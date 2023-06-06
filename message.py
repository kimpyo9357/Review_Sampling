import pika, sys, os,json, threading
import time
import crawling
import combine
import bert
import mysql_command

#connection = pika.BlockingConnection(pika.ConnectionParameters(host='14.47.104.94'))
#channel = connection.channel()

child_thread = True
db = mysql_command.mysql

def main():
    global child_thread
    print(' [*] Waiting for messages. To exit press CTRL+C')
    while(True):
        #channel.queue_declare(queue='spring.python.product') # spring -> python
        #channel.queue_declare(queue='python.spring.product')

        #channel.basic_consume(queue='spring.python.product', on_message_callback=callback, auto_ack=True)
        if child_thread == 1:
            t = threading.Thread(target=read_instruction) ## 스레드 실행 확인 필요
            t.start()
            child_thread -= 1
        time.sleep(5)
        #channel.start_consuming()

def callback(ch, method, properties, body):
    global child_thread
    child_thread = True
    '''with open('./data.json','r', encoding='UTF8') as file:
        data = file.read()
    channel.basic_publish(exchange='', routing_key='python.spring.product', body=data)
    print(" [x] Sent 'Hello World!'")'''
    print("callback")
    print(body)

def read_instruction():
    try:
        with open('./instruction.json', 'r', encoding='UTF8') as file:
            # json_data = file.read()
            json_data = json.load(file)
            print(json_data['instruction'])
            print(json_data['type'])
            if (json_data['instruction'] == 'crawling' and json_data['type'] == 'search'):
                crawling.list_crawling(json_data['data'])  ## 파일 리스트 출력 -> data.json
                # channel.basic_publish(exchange='', routing_key='python.spring.product', body="list")  # list 반환
            elif (json_data['instruction'] == 'crawling' and json_data['type'] == 'review'):
                review = dict()
                pcode = crawling.ret_pcode(json_data['data'])
                crawling.find_review(pcode)
                print("pass crawling " + str(pcode))
                komoran, pre_text = combine.komoran_parsing('reveiew_' + str(pcode) + '.txt')
                print("pass morphological analysis " + str(pcode))
                key, komoran, pre_text = combine.textrank_list(komoran, pre_text)
                print("pass key analysis " + str(pcode))
                pros_cons = bert.BERT_list(key, komoran)
                review['pros'] = dict(pros_cons[-3:])
                review['cons'] = dict(pros_cons[:3])
                with open('./pros_cons.json', 'w', encoding='UTF-8') as make_file:
                    json.dump(review, make_file, indent="\t",ensure_ascii=False)
                # channel.basic_publish(exchange='', routing_key='python.spring.product', body="review")  # review 반환
    except:
        print(' [*] Doesn\'t have instruction or not match type')
        #channel.basic_publish(exchange='', routing_key='python.spring.product', body="except") # 명령 파일 없음
    return

if __name__ == '__main__':
    main()