'''import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

Google_SEARCH_ENGINE_ID =   # Search Engine ID 
Google_API_KEY =  # Custom Search Engine API KEY 
query = "Python review"  # 검색할 쿼리
start_page = "1" # 몇 페이지를 검색할 것인지. 한 페이지 당 10개의 게시물을 받아들일 수 있습니다. 

url = f'https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={query}&start=1&num=1'


response = requests.get(url).json()
Trash_Link = ["tistory", "kin", "youtube", "blog", "book", "news", "dcinside", "fmkorea", "ruliweb", "theqoo", "clien", "mlbpark", "instiz", "todayhumor"] 

search_items = response.get("items")
print(response)


def Google_API(query, wanted_row):
    """
    input : 
        query : str  검색하고 싶은 검색어 
        wanted_row : str 검색 결과를 몇 행 저장할 것인지 
    output : 
        df_google : dataframe / column = title, link,description  
        사용자로 부터 입력받은 쿼리문을 통해 나온 검색 결과를 wanted_row만큼 (100행을 입력받았으면) 100행이 저장된 데이터 프레임을 return합니다.
    """

    print(query)
    query= query.replace("|","OR") #쿼리에서 입력받은 | 기호를 OR 로 바꿉니다 
    #query += "-filetype:pdf" # 검색식을 사용하여 file type이 pdf가 아닌 것을 제외시켰습니다 
    query += " review" # 검색식을 사용하여 file type이 pdf가 아닌 것을 제외시켰습니다
    print(query)
    start_pages=[] # start_pages 라는 리스트를 생성합니다. 

    df_google= pd.DataFrame(columns=['Title','Link','Description']) # df_Google이라는 데이터 프레임에 컬럼명은 Title, Link, Description으로 설정했습니다.

    row_count =0 # dataframe에 정보가 입력되는 것을 카운트 하기 위해 만든 변수입니다. 


    for i in range(1,wanted_row+1000,10):
        start_pages.append(i) #구글 api는 1페이지당 10개의 결과물을 보여줘서 1,11,21순으로 로드한 페이지를 리스트에 담았습니다. 

    for start_page in start_pages:
      # 1페이지, 11페이지,21페이지 마다, 
        url = f"https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={query}&start={start_page}"
        # 요청할 URL에 사용자 정보인 API key, CSE ID를 저장합니다. 
        data = requests.get(url).json()
        # request를 requests 라이브러리를 통해서 요청하고, 결과를 json을 호출하여 데이터에 담습니다.
        search_items = data.get("items")
        # data의 하위에 items키로 저장돼있는 값을 불러옵니다. 
        # search_items엔 검색결과 [1~ 10]개의 아이템들이 담겨있다.  start_page = 11 ~ [11~20] 
        try:
          #try 구문을 하는 이유: 검색 결과가 null인 경우 link를 가져올 수가 없어서 없으면 없는대로 예외처리
            for i, search_item in enumerate(search_items, start=1):
              # link 가져오기 
                link = search_item.get("link")
                if any(trash in link for trash in Trash_Link):
                  # 링크에 dcinside, News 등을 포함하고 있으면 데이터를 데이터프레임에 담지 않고, 다음 검색결과로 
                    pass
                else: 
                    # 제목저장
                    title = search_item.get("title")
                    # 설명 저장 
                    descripiton = search_item.get("snippet")
                    # df_google에 한줄한줄 append 
                    df_google.loc[start_page + i] = [title,link,descripiton] 
                    # 저장하면 행 갯수 카운트 
                    row_count+=1
                    if (row_count >= wanted_row) or (row_count == 300) :
                      #원하는 갯수만큼 저장끝나면
                        return df_google
        except:
          # 더 이상 검색결과가 없으면 df_google 리턴 후 종료 
            return df_google

    
    return df_google

def final(query,wanted_row=100):
    df_google = Google_API(query,wanted_row)
    df_google['search_engine']='Google'
    # 서치엔진 구글을 통해 얻었음을 기록 
    #df_naver = Naver_API(query,wanted_row)
    #df_naver['search_engine']='Naver'
    # 서치엔진 기록 
    #df_daum = Daum_API(query,wanted_row)
    #df_daum['search_engine']='Daum
    df_final= df_google    #pd.concat([df_google,df_naver,df_daum])
    # 전체 네이버 구글 카카오에서 크롤링한 내용을 수평으로 결합 rowbind 
    df_final['search_date'] = datetime.datetime.today
    # 검색한 날자 저장 
    df_final.reset_index(inplace=True,drop=True)
    #인덱스 초기화 
    return df_final

#print(final("Python"))
''' #구글 api 활용 주소 추출
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions 

import time

import pandas as pd
import numpy as np
import json


ser = Service('../chromedriver/chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)


url = 'https://prod.danawa.com/list/?cate=112760'
driver.get(url)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
prod_items = soup.select('div.main_prodlist > ul.product_list > li.prod_item')

with open("data.json",'w',encoding='utf-8') as f:
    for i in range(len(prod_items)-1): #카테고리 출력
        data = {}
        prod_items[i].select('a')[0].text
        title = prod_items[i].select('p.prod_name > a')[0].text
        link = prod_items[i].select('p.prod_name > a')[0].attrs['href']
        spec_list = prod_items[i].select('div.spec_list')[0].text.strip()
        price = prod_items[i].select('li.rank_one > p.price_sect > a > strong')[0].text.strip().replace(',', "")
        data['name'] = title.strip()
        data['price'] = price
        data['category'] = spec_list.split("/")[0]
        #print(title.strip(), spec_list.split("/")[0], price,link, sep = '\n')
        #print(data)
        json.dump(data,f,ensure_ascii=False, indent='\t')
        #print(title.strip(), link)

'''prod_items[0].select('a')[0].text #한개만 테스트
title = prod_items[0].select('p.prod_name > a')[0].text
link = prod_items[0].select('p.prod_name > a')[0].attrs['href']
#spec_list = prod_items[i].select('div.spec_list')[0].text.strip()
#price = prod_items[i].select('li.rank_one > p.price_sect > a > strong')[0].text.strip().replace(',', "")
#print(title, spec_list, price, sep = '   |||  ')
'''

#driver.get(link) 페이지 이동
# 카테고리 검색


#################################################### 제품 크롤링
'''
LOADING_WAIT_TIME= 1
# 크롤링할 상품 코드
pcodes = ['13276568']
# 결과 리스트
result = []


def init_driver():
    options = webdriver.ChromeOptions()
    #options.add_argument("headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)
    return driver

def find_review(pcode, driver):
    file_name = f'reveiew_{pcode}.txt'
    txt_file = open(file_name,'w')
    
    url = f'http://prod.danawa.com/info/?pcode={pcode}'
    driver.get(url)
    # 제휴 쇼핑몰 리뷰 클릭
    WebDriverWait(driver, LOADING_WAIT_TIME).until(
        EC.presence_of_element_located(
            (By.XPATH,'//*[@id="danawa-prodBlog-productOpinion-button-tab-companyReview"]')
        )
    )
    time.sleep(3)
    while True:
        try:
            driver.find_element(By.XPATH,'//*[@id="danawa-prodBlog-productOpinion-button-tab-companyReview"]').click()
            break
        except exceptions.StaleElementReferenceException.e:
            time.sleep(1)
        
    for i in range(1, 10):
        print(str(i) + "번째")
        # 이번 페이지의 모든 리뷰가 로드 될때까지 기다림
        WebDriverWait(driver, LOADING_WAIT_TIME).until(
            EC.visibility_of_all_elements_located(
                (By.CLASS_NAME,'atc')
            )
        )
        for review in driver.find_elements(By.CLASS_NAME,'atc'):
            #result.append(review.text)
            txt_file.write(review.text + '\n')


        if i % 10 == 0:
            # 10개를 다 봤다면 다음 10개를 보는 버튼 클릭, 만약 클릭이 안되면 종료
            try:
                right_btn = driver.find_element(By.XPATH, '//*[@class="mall_review"]//span[@class="point_arw_r"]')
                if right_btn.value_of_css_property('cursor') == 'pointer':
                    right_btn.click()
                else:
                    break
            except exceptions.StaleElementReferenceException.e:
                print(e)
                pass

        else:
            try: # 다음 페이지가 없을 경우
                WebDriverWait(driver, LOADING_WAIT_TIME).until(
                        EC.presence_of_element_located(
                            (By.XPATH, f'//*[@id="danawa-prodBlog-companyReview-content-list"]/div/div/div/a[text()={i + 1}]')
                        )
                    )
            except:
                break
            try:
                driver.find_element(By.XPATH, f'//*[@id="danawa-prodBlog-companyReview-content-list"]/div/div/div/a[text()={i + 1}]').click()
            except exceptions.StaleElementReferenceException.e:
                print(e)
                pass
    txt_file.close()


if __name__ == "__main__":
    driver = init_driver()
    for pcode in pcodes:
        find_review(pcode, driver)
        #for r in result:
            #print(r)
'''