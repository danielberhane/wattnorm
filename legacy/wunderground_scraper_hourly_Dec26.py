
# Daniel Berhane 2025

from bs4 import BeautifulSoup as BS
from selenium import webdriver
from functools import reduce
import pandas as pd
import time
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  
import time
import sys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
from selenium import webdriver


# Set the absolute path to chromedriver
chromedriver_path = './chromedriver'


def render_page(url):
    """Given a url, render it with chromedriver and return the html source"""
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Use ChromeDriverManager to automatically get the correct driver
    service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        return driver.page_source
    finally:
        driver.quit()



def hourly_scraper(page,date_list,type):
    output = pd.DataFrame()

    start_date = datetime.strptime(date_list[0], '%y-%m-%d').date()
    end_date =  datetime.strptime(date_list[1], '%y-%m-%d').date()

    while (start_date != end_date): 
            
            
        url = str(str(page) + str(start_date))

        r = render_page(url)

        soup = BS(r, "html.parser",)
        container = soup.find('lib-city-history-observation') 
        check = container.find('tbody')

        data = []
        data_hour = [] 
        
        for i in check.find_all('span', class_='ng-star-inserted'):
            trial = i.get_text()
            data_hour.append(trial)

        for i in check.find_all('span', class_='wu-value wu-value-to'):
            trial = i.get_text()
            data.append(trial)


        numbers = pd.DataFrame([data[i:i+7] for i in range(0, len(data), 7)],columns=["Temperature","Dew Point","Humidity","Wind Speed","Wind Gust","Pressure","Precipitation"])
        hour = pd.DataFrame(data_hour[0::17],columns=["Time"])
        wind = pd.DataFrame(data_hour[7::17],columns=["Wind"])
        condition = pd.DataFrame(data_hour[16::17],columns=["Condition"])

        dfs = [hour,numbers,wind,condition]

        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
        df_final['Date'] = str(start_date)

        output = output.append(df_final)
        print(str(str(start_date) + ' finished!'))

        start_date  = start_date + timedelta(days=1)

    return output


def output_to_file (dates_list, page="https://www.wunderground.com/history/daily/us/ny/ithaca/KITH/date/"):
    hourly = hourly_scraper(page,dates_list,"C")
    return (hourly)

dates = ['15-1-1','16-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2015.csv')

dates = ['16-1-2','17-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2016.csv')

dates = ['17-1-1','18-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2017.csv')

dates = ['18-1-2','19-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2018.csv')

dates = ['19-1-2','20-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2019.csv')

dates = ['20-1-2','21-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2020.csv')

dates = ['21-1-2','22-1-1']
df_output = output_to_file (dates)
df_output.to_csv ('Year 2021.csv')
print (df_output)


