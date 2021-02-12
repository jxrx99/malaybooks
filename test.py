# https://github.com/yangobeil/book-recommendation
# https://medium.com/swlh/how-simple-is-it-to-build-an-end-to-end-item-based-recommender-system-90f6d959e7c2
import time
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# url = 'https://www.bookurve.com/book/9789672328001/di-situ-langit-dijunjung/'
# url = 'https://www.bookxcessonline.com/collections/malay-workbooks/products/kunci-a-tahun-6-9789674555542'

def get_soup(url):
    # Instantiate an Options object
    # and add the “ — headless” argument
    opts = Options()
    opts.add_argument('--headless')

    # Set the location of the webdriver
    path = r'/Users/jufri/Google Drive/ML_project/MalayBooks/chromedriver'    
    
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
    
    print("Book info fn:", url)

    soup = get_soup(url)
    
    data = {
        'title': soup.find('h1', {'class': 'product_name'}).text,
        'author': (soup.find('a', {'class': 'author_search'}).text.replace('\n', '')).strip(),
        'ISBN':  soup.find('td', {'class': 'second'}).text,
        'summary': soup.find('div', {'class': 'description bottom'}).text.strip(),
        'image': soup.find('img', {'class': 'none lazyautosizes ls-is-cached lazyloaded'})['src']
	    # url = 'https://www.bookxcessonline.com/collections/malay-workbooks/products/kunci-a-tahun-6-9789674555542'

        #     url = 'https://www.bookurve.com/book/9789672328001/di-situ-langit-dijunjung/'
        #    'title': soup.find('h1', {'class': 'product_name'}).text,
        #     'author': soup.find('a', {'class': 'author_search'}).text,
        #     'image': soup.find('img', {'class': 'au-target'})['src']

        # 'title': soup.find('h1', {'class': 'book-title'}).text,
        # 'author': soup.find('a', {'class': 'book-meta-author-name'}).text.replace('\n', ''),
        # 'ISBN':  soup.find('div', {'id': 'sel-buy-box'}).find('span', {'class': 'buy-box--isbn'}).text[6:],
        # 'summary': soup.find('div', {'class': 'book-summary'}).text.replace('\n', ''),
        # 'image': soup.find('img', {'class': 'book-image'})['src']
    }
    

    
    return data

import tensorflow_hub as hub
import tensorflow_text

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

def get_links_in_page(category_list, full_data):
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


    # what = soup.find_all('a', {'class': 'hidden-product-link'})[1]
    # what['href']

    # remove books appearing more than once
    book_data_no_duplicates = []
    titles = []
    for book in full_data:
        if book['title'] not in titles:
            titles.append(book['title'])
            book_data_no_duplicates.append(book)
            
    full_date = book_data_no_duplicates

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    # add vectors to the data
    for element in full_data:
        element['vector'] = embed(element['summary'])[0]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

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

# get_soup(url)
# get_book_info(url)
get_links_in_page(category_list, full_data)