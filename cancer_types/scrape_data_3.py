import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the driver (example with Chrome)
driver = webdriver.Chrome()

# Open the webpage
driver.get("https://www.mdanderson.org/patients-family/diagnosis-treatment/cancer-types.html")

# List of letters to iterate through
letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Q', 'U', 'X', 'Y', 'Z']]

# Initialize a dictionary to hold the links
links_dict = {}

# Iterate through each letter
for letter in letters:
    links_dict[letter] = []  # Initialize an empty list for this letter
    try:
        # Scroll to each letter section
        letter_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, letter))
        )
        ActionChains(driver).move_to_element(letter_section).perform()

        # Wait for the first link in this section to be visible
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, f"#{letter} .glossary-search-result a"))
        )

        # Extract all links within the letter section
        links = driver.find_elements(By.CSS_SELECTOR, f"#{letter} .glossary-search-result a")
        for link in links:
            links_dict[letter].append(link.get_attribute('href'))

    except Exception as e:
        print(f"Error processing letter {letter}: {e}")

# Close the driver after the operation is complete
driver.quit()

file_path = 'cancer_types_links.json'

# Writing the JSON data to a file
with open(file_path, 'w') as json_file:
    json.dump(links_dict, json_file, indent=4)

file_path
