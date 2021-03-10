import time
import requests
import tensorflow_hub as hub
import tensorflow_text
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_soup(url):
    """ Loads a dynamic page through selenium and get Beautiful soup to get the page source.

    Args:
        url (string): url of page to be scrapped

    Returns:
        BeautifulSoup: page source of the url being loaded
    """
    # Instantiate an Options object
    # and add the “ — headless” argument
    opts = Options()
    opts.add_argument('--headless')

    # Set the location of the webdriver
    path = r'./chromedriver'    
    
    # Instantiate a webdriver
    driver = webdriver.Chrome(options=opts,executable_path = path)

    time.sleep(2)

    # Load the HTML page
    driver.get(url)

    # Put the page source into a variable and create a BS object from it
    soup_file=driver.page_source
    soup = BeautifulSoup(soup_file)

    # Close the browser
    driver.quit()

    return soup

def get_book_info(url):
    """ Retrieve the individual book information

    Args:
        url (string): hyperlink of individual book to be scrapped

    Returns:
        dictionary: contains book information pulled by beautiful soup
    """
    
    print("Book info fn:", url)

    soup = get_soup(url)
    
    data = {
        'title': soup.find('h1', {'class': 'product_name'}).text,
        'author': (soup.find('a', {'class': 'author_search'}).text.replace('\n', '')).strip(),
        'ISBN':  soup.find('td', {'class': 'second'}).text,
        'summary': soup.find('div', {'class': 'description bottom'}).text.strip(),
        'image': soup.find('img', {'class': 'none lazyautosizes ls-is-cached lazyloaded'})['src']
	    # url = 'https://www.bookxcessonline.com/collections/malay-workbooks/products/kunci-a-tahun-6-9789674555542'

    }
    

    
    return data

def get_links_in_page(category_list, full_data):
    """ Retrieve the hyperlink in the page containing individual book links

    Args:
        category_list (list): list of url, preferably search result page to get links of individual books
        full_data (list): full data containing book metadata
    """
    for page in category_list:
        print("Link fn:", page)

        soup = get_soup(page)

        for book in soup.find_all('a', {'class': 'hidden-product-link'}):
            base_url = 'https://www.bookxcessonline.com/'
            url = base_url + book['href']
            try:
                full_data.append(get_book_info(url))
            except Exception as e:
                print(e)

    # remove books appearing more than once
    book_data_no_duplicates = []
    titles = []
    for book in full_data:
        if book['title'] not in titles:
            titles.append(book['title'])
            book_data_no_duplicates.append(book)
            
    full_data = book_data_no_duplicates

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    # add vectors to the data
    for element in full_data:
        element['vector'] = embed(element['summary'])[0]

    vectors = [item['vector'] for item in full_data]
    X = np.array(vectors)

    # calculate similarity based on Euclidean distance
    sim = euclidean_distances(X)
    indices = np.vstack([np.argsort(-arr) for arr in sim])

    # calculate similarity based on cosine distance
    cos_sim = cosine_similarity(X)
    cos_indices = np.vstack([np.argsort(-arr) for arr in cos_sim])

    # find most similar books for each case
    for i, book in enumerate(full_data):
        book['euclidean'] = indices[i][1:21]
        book['cosine'] = cos_indices[i][1:21]

    # remove vectors from dict
    for book in full_data:
        book.pop('vector')

    # full_data[0]

    # save the data
    import pickle

    with open('books.pkl', 'wb') as f:
        pickle.dump(full_data, f)

# url = 'https://www.bookxcessonline.com/collections/malay-workbooks/products/kunci-a-tahun-6-9789674555542'

# get data about most popular books on the website
full_data = []

# page='https://bookxcessonline.myshopify.com/collections/malay-childrens-books?page=3'
category_list = [
    'https://www.bookxcessonline.com/collections/malay-childrens-books?page=1',
    'https://www.bookxcessonline.com/collections/malay-childrens-books?page=2',
    'https://www.bookxcessonline.com/collections/malay-childrens-books?page=3',
    'https://www.bookxcessonline.com/collections/malay-childrens-books?page=4',
    'https://www.bookxcessonline.com/collections/malay-fiction?page=1',
    'https://www.bookxcessonline.com/collections/malay-fiction?page=2',
    'https://www.bookxcessonline.com/collections/malay-non-fiction?page=1',
    'https://www.bookxcessonline.com/collections/malay-non-fiction?page=2',
    'https://www.bookxcessonline.com/collections/malay-non-fiction?page=3',
    'https://www.bookxcessonline.com/collections/malay-spirituality-and-religion?page=1',
    'https://www.bookxcessonline.com/collections/malay-spirituality-and-religion?page=2'
    ]
    
# get_links_in_page(category_list, full_data) # 
# get_soup(url)
# get_book_info(url)
