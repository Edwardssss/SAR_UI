# Import the required Library 
from selenium import webdriver

#FireFox path
driver = webdriver.Chrome(r'C:\Users\Lenovo\Downloads\chromedriver_win32\chromedriver.exe')

# Web URL 
driver.get('https://www.tutorialspoint.com/python') 

element = driver.find_element_by_tag_name('') 

# Get Text 
print(element.text) 

# Close the window 
driver.close() 



## Download Chrome Webdriver from https://chromedriver.storage.googleapis.com
