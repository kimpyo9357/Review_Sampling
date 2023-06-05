import pika, sys, os,json

connection = pika.BlockingConnection(pika.ConnectionParameters(host='14.47.104.94'))
channel = connection.channel()

def main():
    channel.queue_declare(queue='spring.python.product')
    channel.queue_declare(queue='python.spring.product')

    channel.basic_consume(queue='spring.python.product', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

def callback(ch, method, properties, body):
    with open('./data.json','r', encoding='UTF8') as file:
        data = file.read()
    channel.basic_publish(exchange='', routing_key='python.spring.product', body=data)
    print(" [x] Sent 'Hello World!'")

if __name__ == '__main__':
    main()