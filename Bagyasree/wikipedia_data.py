import requests
from bs4 import BeautifulSoup
import wikipedia

base_url = "https://en.wikipedia.org/wiki/"
topic = "mukamba"
topic2 = "motivate employee"

page_bs4 = requests.get(base_url+topic2)
contents = BeautifulSoup(page_bs4.content, 'html.parser')
# print(contents.get_text().encode('utf-8'))
categs = contents.find_all('href')
# print(categs)

suggestions = wikipedia.search(topic2)
print(suggestions)
# print(page.content.encode('utf-8'))

# depth = 1
# parsed_topics = []
# queue = page.links
# while depth:
#     inner_queue = []
#     for link in queue: 
#         if link not in parsed_topics:
#             try: 
#                 inner_page = wikipedia.page(link)
#                 print(inner_page.links)
#                 inner_queue.extend(inner_page.links)
#                 parsed_topics.append(link)
#             except: 
#                 pass #skipping ambiguous terms for now.
#     queue = inner_queue
#     depth -= 1

# print(parsed_topics)

# links = page.links
# categories = set(page.categories)
# categories_list = list(categories)
# print("Categories:",categories)
# print(wikipedia.search(topic2))
# print("Links:",links)
# not_relevant = 0
# for link in links:
#     try:
#         print("Topic:",link, " - ", end = ' ')
#         linked_page = wikipedia.page(link)
#         common = list(set(linked_page.categories) & categories)
#         if len(common):
#             print("Relevant. COMMON: ",common)
#         else:
#             not_relevant += 1
#             print("Not relevant.")
#     except:
#         print("in except")
#         pass

# print("Links:",len(links))
# print("Not relevant:",not_relevant)



