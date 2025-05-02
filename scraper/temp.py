from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import requests
import os

driver = webdriver.Chrome()

url = "https://www.scobserver.in/cases/?fwp_case_status=decided"
driver.get(url)

time.sleep(5)

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

soup = BeautifulSoup(driver.page_source, 'html.parser')

case_links = []
for link in soup.find_all('h2'):
    href = link.find('a')['href'] if link.find('a') else None
    if href:
        case_links.append(href)

driver.quit()

for case_link in case_links:
    print(case_link)

for case_link in case_links:
    print(f"Visiting: {case_link}")
    response = requests.get(case_link)
    case_soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the <a> tag with class 'single-cases__judgment'
    judgment_link = case_soup.find('a', class_='single-cases__judgment')
    if judgment_link and 'href' in judgment_link.attrs:
        pdf_url = judgment_link['href']
        print(f"Downloading PDF: {pdf_url}")
        
        # Download the PDF
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code == 200:
            pdf_name = os.path.join("./judgments/", pdf_url.split('/')[-1])
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)
            print(f"Saved: {pdf_name}")
        else:
            print(f"Failed to download: {pdf_url}")
    else:
        print("Judgment PDF not found on the page.")