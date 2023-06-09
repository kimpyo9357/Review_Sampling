import pymysql

# MySQL 데이터베이스에 연결

class mysql:
    def __init__(self):
        self.db = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='password',
            database='db'
        )
        # 커서 개체 만들기
        self.cursor = self.db.cursor()

        # 데이터 정의
        check = "SHOW TABLES like 'product_data'"
        self.cursor.execute(check)
        result = self.cursor.fetchall()
        if len(result) == 0:
            self.cursor.execute("CREATE TABLE product_data (id INT NOT NULL PRIMARY KEY, name VARCHAR(20) NOT NULL, category VARCHAR(10))")
            print("make product_data table")


        check = "SHOW TABLES like 'productdetail_data'"
        self.cursor.execute(check)
        result = self.cursor.fetchall()
        if len(result) == 0:
            self.cursor.execute("CREATE TABLE productdetail_data (id INT NOT NULL, name VARCHAR(20) NOT NULL, descr VARCHAR(1000), PRIMARY KEY(id), FOREIGN KEY(id) REFERENCES product_data (id))")
            print("make productdetail_data table")

        check = "SHOW TABLES like 'productanal_data'"
        self.cursor.execute(check)
        result = self.cursor.fetchall()
        if len(result) == 0:
            self.cursor.execute("CREATE TABLE productanal_data (id INT NOT NULL, name VARCHAR(20) NOT NULL, pro1 VARCHAR(20), pro2 VARCHAR(20), pro3 VARCHAR(20), con1 VARCHAR(20), con2 VARCHAR(20), con3 VARCHAR(20), PRIMARY KEY(id), FOREIGN KEY(id) REFERENCES product_data (id))")
            print("make productanal_data table")

    def insert(self,tables,data = []):
        com = 0
        result = self.search(tables,data[0])
        if len(result) == 0:
            if (tables == 'product' and len(data) == 3):
                SQL = "INSERT INTO product_data VALUES " +str(tuple(data))
                com = 1
            elif(tables == 'productdetail' and len(data) == 4):
                SQL = "INSERT INTO productdetail_data VALUES "+str(tuple(data))
                com = 1
            elif(tables == 'productanal' and len(data) == 8):
                SQL = "INSERT INTO productanal_data VALUES "+str(tuple(data))
                com = 1
            if (com == 1):
                self.cursor.execute(SQL)
                self.db.commit()
        else:
            print("Data already exists.")

    def search(self,tables, data = '*'):
        com = 0
        if data == '*':
            if (tables == 'product'):
                SQL = "SELECT * FROM product_data"
                com = 1
            elif(tables == 'productdetail'):
                SQL = "SELECT * FROM productdetail_data"
                com = 1
            elif(tables == 'productanal'):
                SQL = "SELECT * FROM productanal_data"
                com = 1
        else:
            if (tables == 'product'):
                SQL = "SELECT * FROM product_data WHERE id ='" + str(data) +"'"
                com = 1
            elif(tables == 'productdetail'):
                SQL = "SELECT * FROM productdetail_data WHERE id ='" + str(data) +"'"
                com = 1
            elif(tables == 'productanal'):
                SQL = "SELECT * FROM productanal_data WHERE id ='" + str(data) +"'"
                com = 1
        if (com == 1):
            self.cursor.execute(SQL)
            data = self.cursor.fetchall()
            return data[0]

    def delete(self,tables,data):
        com = 0
        result = self.search(tables,data)
        if len(result) != 0:
            if (tables == 'product'):
                SQL = "DELETE FROM product_data WHERE id =" + str(data)
                com = 1
            elif(tables == 'productdetail'):
                SQL = "DELETE FROM productdetail_data WHERE id =" + str(data)
                com = 1
            elif(tables == 'productanal'):
                SQL = "DELETE FROM productanal_data WHERE id =" + str(data)
                com = 1
            if (com == 1):
                self.cursor.execute(SQL)
                self.db.commit()
        else:
            print("Data doesn't exist.")

def main():
    print ("Main Function")
    db = mysql()
    db.insert('product',[1,2,3])
    db.search('product',1)
    '''mysql.cursor.close()
    mysql.db.close()'''
    

if __name__ == "__main__":
	main()

