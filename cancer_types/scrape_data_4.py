from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize the WebDriver (adjust the path to your chromedriver)
driver = webdriver.Chrome()

# Open the page
driver.get('https://www.dana-farber.org/cancer-care/types?rows=150')

# Wait for the elements to load
time.sleep(5)  # Adjust the sleep time based on your connection speed and page load time

# Find all link elements inside cancer-type-details
cancer_links = driver.find_elements(By.CSS_SELECTOR, 'cancer-type-details a[href*="/cancer-care/types/"]')

# Extract the href attributes
cancer_types = []
for link in cancer_links:
    href = link.get_attribute('href')
    if href not in cancer_types:  # Avoid duplicates
        cancer_types.append(href)

print(cancer_types)

# Close the driver
driver.quit()
