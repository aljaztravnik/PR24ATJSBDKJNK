"""from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


service = Service()
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv


service = Service(executable_path='C:/Users/janst/programs/chromedriver-new/chromedriver-win64/chromedriver.exe')  # Specify the path to chromedriver
options = Options()
driver = webdriver.Chrome(service=service, options=options)
url = "https://en.wikipedia.org/wiki/Aeroflot_accidents_and_incidents_in_the_1970s"
driver.get(url)

table = driver.find_element(By.TAG_NAME, "table")
rows = table.find_elements(By.TAG_NAME, "tr")
data = []


for row in rows:
    cells = row.find_elements(By.TAG_NAME, "td")
    row_data = [cell.text for cell in cells]
    data.append(row_data)

csv_file_path = "./podatki/aeroflot_accidents_1970s.csv"

columns = ["Date", "Location","Aircraft", "Tail number", "Airline division", "Aircraft damage", "Fatalities", "Description", "Refs"]

with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    writer.writerows(data)


driver.quit()
