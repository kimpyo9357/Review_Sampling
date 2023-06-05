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
import re
import json

def list_crawling(code,driver):
    #url = 'https://prod.danawa.com/list/?cate=112760'
    url = 'https://search.danawa.com/dsearch.php?k1='+str(code)
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    prod_items = soup.select('div.main_prodlist > ul.product_list > li.prod_item')

    with open("data.json",'w',encoding='utf-8') as f:
        for i in range(len(prod_items)-1): #카테고리 출력
            data = {}
            try:
                title = prod_items[i].select('p.prod_name > a')[0].text
                link = prod_items[i].select('p.prod_name > a')[0].attrs['href']
                spec_list = prod_items[i].select('div.spec_list')[0].text.strip()
                price = prod_items[i].select('li.rank_one > p.price_sect > a > strong')[0].text.strip().replace(',', "")
                data['name'] = title.strip()
                data['price'] = price
                data['category'] = spec_list.split("/")[0]
                print(title.strip(), spec_list.split("/")[0], price,link, sep = '\n')
                #print(data)
                json.dump(data,f,ensure_ascii=False, indent='\t')
                #print(title.strip(), link)
            except:
                pass
'''prod_items[0].select('a')[0].text #한개만 테스트
title = prod_items[0].select('p.prod_name > a')[0].text
link = prod_items[0].select('p.prod_name > a')[0].attrs['href']
#spec_list = prod_items[i].select('div.spec_list')[0].text.strip()
#price = prod_items[i].select('li.rank_one > p.price_sect > a > strong')[0].text.strip().replace(',', "")
#print(title, spec_list, price, sep = '   |||  ')
'''

#driver.get(link) 페이지 이동
# 카테고리 검색
pcodes = ['13276568']


#################################################### 제품 크롤링
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)
    return driver

def find_review(pcode, driver):
    LOADING_WAIT_TIME = 1
    # 크롤링할 상품 코드

    # 결과 리스트
    result = []
    file_name = f'reveiew_{pcode}.txt'
    txt_file = open(file_name,'w', encoding="UTF-8")

    url = f'http://prod.danawa.com/info/?pcode={pcode}'
    driver.get(url)
    # 제휴 쇼핑몰 리뷰 클릭
    WebDriverWait(driver, LOADING_WAIT_TIME).until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="danawa-prodBlog-productOpinion-button-tab-productOpinion"]')
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

def ret_pcode(url):
    tag = url.split("/")[-1][1:]
    pcode = re.sub(r'[^0-9]', '', tag.split("&")[0])
    return pcode

if __name__ == "__main__":
    driver = init_driver()
    #list_crawling("4090",driver)
    pcode = ret_pcode("https://prod.danawa.com/info/?pcode=17982347&keyword=4090&cate=112753")
    find_review('7097410', driver)
    '''for pcode in pcodes:
        find_review(pcode, driver)
        #for r in result:
            #print(r)'''