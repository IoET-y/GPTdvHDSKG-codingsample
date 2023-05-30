#----------------------------------------------------------#
#            A key words' website links collector based on GPT-API via prompting
#----------------------------------------------------------#
import openai
import os
OPENAI_API_KEY="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"
import pandas as pd
import re

# Authenticate the API
openai.api_key = OPENAI_API_KEY #api_key
def split_sentences(text):
    # 使用正则表达式分割句子，忽略数字.数字格式的内容
    #sentences = re.findall(r'(?<!\d)\s*([^.!?]+[.!?])\s*(?!\d)', text)
    #sentences = re.split(r'(?<!\d)[.!?](?!\d)', text)
    sentences = re.split(r'\n', text)
    # 移除前后的空格
    sentences = [sentence.strip() for sentence in sentences if not not sentence.strip()]
    return sentences

def ask_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate GPT engine
        prompt=prompt,
        max_tokens=3500,  # Limit the response length
        n=5,  # Number of responses to generate
        stop=None,  # Stop sequence for the response
        temperature=1.0,  # Sampling temperature (higher values give more random outputs)
        top_p=1,  # Top-p sampling
    )

    answer = response.choices[0].text.strip()  # Extract the answer text
    return answer

# def main(text):
#      question = input("Ask your question (type 'quit' to exit): ")
#        if question.lower() == "quit":
#            break
#        prompt = f"Answer the following question: {question}"
#        answer = ask_gpt(prompt)
#        print(f"Answer: {answer}")

if __name__ == "__main__":
    # file = open('allSent_test2.txt')
    GPT_links = []   # this list will be classified via our bert classifier to figure out those about The key words we care about, e.g. Hypertension, hypercholesterol, diabetes:
    # file_w = open('allGPT_website.txt', 'w')

    #question = 'use a Simple format to separately list the entities and direct relationships from this sentence:'+line
    question = 'Please help me search another 15 website pages, which must include the context about hypertension(e.g. what is hypertension, hypertension\'s definition, symptom, etc.). For example, you can find websites like this given website: https://en.wikipedia.org/wiki/Hypertension, which gives contents：“Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.” Please give me answers in this format: 1. https://www.healthline.com/health/high-blood-pressure-hypertension'
               #'without any other context!'
    prompt = f"{question}"
    # answer = ask_gpt(prompt)
    # print(f"Answer: {answer}")
    # prompt_history = prompt
    # prompt_history = prompt_history + '\n anwser is '+ answer
    df = pd.read_excel('/Users/godspeed/Desktop/AEI_Paper_Yangdi/HDSKG_SANER/web_classifier/GPT_links.xlsx')
    # file_w = open('GPT_scrapy_links.txt', 'w')
# Print the dataset
#     print(df.head())
    for i in range(0,50):
        answer = ask_gpt(prompt)
        print(f"{answer}")
        answer = re.sub(r"\d+\. ", '', answer)
        answer = re.sub(r"\d+\.", '', answer)
        answer = split_sentences(answer)
        for j in answer:
            if j in GPT_links:
                continue
            else:
                GPT_links.append(j)
        # print(links)
        if len(GPT_links) == 300:
            break
        # file_w.write(answer + '\n')
    # file_w.close()
#     id +=1
#  Create a new DataFrame with the website texts and labels
    df['website_links'] = GPT_links
    output_file_path = '/Users/godspeed/Desktop/AEI_Paper_Yangdi/HDSKG_SANER/web_classifier/GPT_links.xlsx'
    df.to_excel(output_file_path, index=False)

#----------------------------------------------------------#
#           start processing training set
#----------------------------------------------------------#

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import string
import random
import nltk
from nltk.corpus import stopwords

# Download the stopwords if you haven't already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import wordninja
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
import wordninja
#
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

#java format webpage scraper
#get java page
def java_enable_get_page(website_link):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36")
    # Set up the Chrome driver (replace "path_to_chromedriver" with your chromedriver path)
    webdriver_service = Service('path_to_chromedriver')
    # Instantiate Chrome webdriver
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    # Specify the URL of the website you want to scrape
    # Open the website in Chrome
    driver.get(website_link)
    # Get the page source after JavaScript execution
    page_source = driver.page_source
    # Process the page source using your preferred parsing library (e.g., BeautifulSoup)
    # ...
    # if page_source.status_code >= 200 and page_source.status_code <= 299:
    soup = BeautifulSoup(page_source, 'html.parser')
    text = soup.get_text()
    print(text)
    # else:
    # print('Error:', response.status_code)
    #
    # Quit the webdriver
    driver.quit()
    return text

# get html web
def get_website_text(website_link):
    proxies = [{'http': 'http://165.154.236.214:80', 'http': 'http://34.126.187.77:80'},{'http': 'http://206.189.146.13:8080', 'http': 'http://43.156.78.106:1080'},{'http': 'http://47.74.152.29:8888', 'http': 'http://74.138.24.67:8080'}, {'http': 'http://139.99.77.57:8118', 'socks5': 'socks5://28.199.128.10:59166'}]
    user_agents = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36','Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16','Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50']
    proxy = random.choice(proxies)
    hd = {'User-Agent': random.choice(user_agents)}
    try:
        response = requests.get(website_link,headers=hd)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()#separator=''
        word =  word_tokenize(text)
        num = len(word)
        if "enable javascript" in text or "cookies to continue" in text:
            text = java_enable_get_page(website_link)

        elif num > 50:
            text = text

        else:
            hd = {'User-Agent': random.choice(user_agents)}
            response = requests.get(website_link,headers=hd)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()#separator='
            tmp =  word_tokenize(text)
            if len(tmp) <10:
                text = 'sorry the website you search is null!, '
        return text
    except:
        return ""

## ----------------
#      pre-process web content
##----------------
def split_sentences(text):
    # 使用正则表达式分割句子，忽略数字.数字格式的内容
    #sentences = re.findall(r'(?<!\d)\s*([^.!?]+[.!?])\s*(?!\d)', text)
    #sentences = re.split(r'(?<!\d)[.!?](?!\d)', text)
    sentences = re.split(r'(?<!\d\.)[.!?]\s+|\n', text)

    # 移除前后的空格
    sentences = [sentence.strip() for sentence in sentences if not not sentence.strip()]
    return sentences

# split word, reform sentences
def deconnect_text(input_text):
    split_words = wordninja.split(input_text)
    cleaned_words = []

    for word in split_words:
        # Capitalize the first letter of each word
        cleaned_word = word.capitalize()
        cleaned_words.append(cleaned_word)

    # Join the cleaned words with spaces
    cleaned_sentence = ' '.join(cleaned_words)
    return cleaned_sentence
# clean text

def clean_text(sentences):
    clean_sent = []
    splited_sent = []
    # id_tag = 0
    # sentences = clean_non_format(sentences)
    for text in sentences:
        text_tmp = clean_non_format(text)
        # id_tag +=1
        # print(str(id_tag)+' '+text)
        #Convert all the text to lowercase
        #remove non-eng sentences
        text = clean_non_eng(text_tmp)
        text = text.lower()
        text = is_hypertension(text)
        if text == "no":
            continue
        else:
            text = text
        #Remove punctuation
        #text = text.translate(str.maketrans('', '', string.punctuation))
        #Remove numbers
        text = text.translate(str.maketrans('', '', string.digits))
        #replace_abbreviations

        text = text.replace('i.e.', 'that is')
        text = text.replace('e.g.', 'for example')
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        a =  word_tokenize(text)
        num = len(a)
        if num < 3:
            continue
        else :
            # clean 语气词和助动词
            #pattern = r'\b(a|an|the|am|is|are|was|were|be|being|been|have|has|had|do|does|did|may|might|must|shall|should|will|would|can|could)\b'
            #pattern = r'\[(?<!\d)\]'
            #cleaned_sentence = re.sub(pattern, '', text)
            # clean特殊字符
            cleaned_sentence = re.sub(r'^(?<!\d)', '', text)
            cleaned_sentence = re.sub(r'doi(?<!\d)', '', cleaned_sentence)
            cleaned_sentence = re.sub(r'[^\w\s]', '', cleaned_sentence)
            cleaned_sentence = re.sub(r'http\S+', '', cleaned_sentence)
            #cleaned_sentence = correct_text(cleaned_sentence)
        # print(cleaned_sentence)
            cleaned_sentence = deconnect_text(cleaned_sentence)
            cleaned_sentence = cleaned_sentence.lower()
            cleaned_sentence = cleaned_sentence + '.'
            clean_sent.append(cleaned_sentence)
        #text = text.replace('[]', '')
        # Remove stop words
        # words = text.split()
        # words = [word for word in words if word not in stop_words]
    text = ' '.join(clean_sent)

    return text

#whether content keyword specially for webclassifier use
def is_hypertension(sentence):

    if "hypertension" in sentence or "high blood" or "pressure" in content:
        text = sentence
    else:
        text = "no"
    return text


def is_english(sentence):
    try:
        language = detect(sentence)
    except:
        return False

    if language == 'en':
        return True
    else:
        return False
# use english web only
def clean_non_eng(text):
    # Tokenize text by sentences
    sentences = sent_tokenize(text)

    # Filter out non-English sentences
    english_sentences = [sentence for sentence in sentences if is_english(sentence)]

    return ' '.join(english_sentences)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def is_sentence_complete(sentence):
    words = word_tokenize(sentence)
    tagged = nltk.pos_tag(words)

    has_noun = False
    has_verb = False
    has_object = False

    for word, tag in tagged:
        wordnet_tag = get_wordnet_pos(tag)
        if wordnet_tag == wordnet.NOUN:
            has_noun = True
        elif wordnet_tag == wordnet.VERB:
            has_verb = True
        elif wordnet_tag == wordnet.ADJ or wordnet_tag == wordnet.ADV:  # considering adjectives/ adverbs as possible objects for simplicity
            has_object = True

    return has_noun and has_verb and has_object

# remove non-formated text
def clean_non_format(text):
    sentences = sent_tokenize(text)

    complete_sentences = [sentence for sentence in sentences if is_sentence_complete(sentence)]

    return ' '.join(complete_sentences)

# for initializing use, you can load your links set for training data with following command
# Load the training use dataset
df = pd.read_excel('/Users/godspeed/Desktop/AEI_Paper_Yangdi/training_link_set.xlsx')
# Print the dataset
print(df.head())

# Convert the website links to text data and preprocess it
website_links = df['website_link'].tolist()
website_texts = []
id = 0

# scrapy the weboage content and store in a dataset for Bert training
for website_link in website_links:
    print(id)
    content = get_website_text(str(website_link))
    sentences = split_sentences(content)
    content = clean_text(sentences)
    website_text =content
    website_texts.append(website_text)
    id +=1 # indicate scrapy process of which pages

# Create a new DataFrame with the website texts and labels
df['website_text'] = website_texts
output_file_path = '/Users/godspeed/Desktop/AEI_Paper_Yangdi/training_text_set.xlsx' # store the links and corresponding page contents in a xlsx file
df.to_excel(output_file_path, index=False)

# load training set for classifier training
train = pd.read_excel('/Users/godspeed/Desktop/AEI_Paper_Yangdi/training_text_set.xlsx')
labels = train['category'].tolist()
texts = train['website_text'].tolist()

#----------------------------------------------------------#
#          Bert phase
#----------------------------------------------------------#

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from torch.utils.data import DataLoader, Dataset

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
NUM_CLASSES = 2
MAX_LEN = 512  # Change this to 1024

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# Load the pre-trained DistilBERT model with a modified configuration
config = DistilBertConfig.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_CLASSES)
model = DistilBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, config=config)
# Update the position embeddings size
model.distilbert.embeddings.position_embeddings = torch.nn.Embedding(MAX_LEN, config.dim)

# Dataset class
class WebsiteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Handle NaN values
        if pd.isna(text):
            text = ""

        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split the dataset into training and temporary sets (70%-30%)
X_train_1, X_temp_1, y_train_1, y_temp_1 = train_test_split(texts[0:100], labels[0:100], test_size=0.3, random_state=22)
X_train_2, X_temp_2, y_train_2, y_temp_2 = train_test_split(texts[101:200], labels[101:200], test_size=0.3, random_state=22)
X_train = X_train_1+X_train_2
y_train = y_train_1+y_train_2
print(y_train)
X_temp = X_temp_1+X_temp_2
y_temp = y_temp_1+y_temp_2
# Split the temporary set into validation and test sets (each 10% of the original dataset)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=22)
# Create WebsiteDataset instances for training, validation, and test sets
train_dataset = WebsiteDataset(X_train, y_train, tokenizer, max_length=MAX_LEN)
val_dataset = WebsiteDataset(X_val, y_val, tokenizer, max_length=MAX_LEN)
test_dataset = WebsiteDataset(X_test, y_test, tokenizer, max_length=MAX_LEN)

# Create data loaders for training, validation, and test sets
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)#, num_workers=4)
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)#, num_workers=4)
test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)#, num_workers=4)

print('dataloader done')
# Training and Evaluation loop


device = torch.device('mps')
model.to(device, dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 10
criterion = torch.nn.CrossEntropyLoss()
print('loop set done')

# Learning rate scheduler
# num_training_steps = len(train_loader) * epochs
# num_warmup_steps = int(num_training_steps * 0.1)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# Early stopping parameters
early_stopping_patience = 2
best_val_loss = float('inf')
patience_counter = 0
print('start epoch')


for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        # scheduler.step()

    avg_train_loss = total_train_loss / len(train_data_loader)

    # Evaluate on validation set
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_val_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    avg_val_loss = total_val_loss / len(val_data_loader)
    val_accuracy = correct_predictions.to(dtype=torch.float32) / len(val_dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}")

    # # Early stopping
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     patience_counter = 0
    #     torch.save(model.state_dict(), 'website_classifier_10.pt')
    # else:
    #     patience_counter += 1
    #     if patience_counter >= early_stopping_patience:
    #         print("Early stopping triggered. Stopping training.")
    #         break
    # Test loop
    model.eval()
    test_correct_predictions = 0

    for batch in test_data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            test_correct_predictions += torch.sum(preds == labels)

    test_accuracy = test_correct_predictions.to(dtype=torch.float32) / len(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
# Save the trained model
torch.save(model.state_dict(), 'website_classifier_2023.pt')


def scrape_website(website_link):
    content = get_website_text(str(website_link))
    sentences = split_sentences(content)
    content = clean_text(sentences)
    website_text =content+'\n'
    return website_text

def encode_text(text, tokenizer, max_length=MAX_LEN):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

def classify_website(model, tokenizer, url, device):
    content = scrape_website(url)
    if "404" in content or "page not found" in content:
        print("The sentence contains either '404' or 'Page Not Found'.")
        predicted_class = 0
    else:
        encoded_text = encode_text(content, tokenizer)
        input_ids = encoded_text['input_ids'].unsqueeze(0).to(device)
        print(input_ids)
        attention_mask = encoded_text['attention_mask'].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            print('the probilities is')
            print(probabilities[0])
            print('the probilities [0][1] is')
            print(probabilities[0][1])
            if probabilities[0][1] > probabilities[0][0]:
                predicted_class = 1
            else:
                predicted_class = 0
            #predicted_class = torch.argmax(probabilities[0])

    return predicted_class


# Load the trained model
model_path = 'website_classifier_2023.pt'
trained_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
trained_model.load_state_dict(torch.load(model_path))
trained_model.to(device)# dtype=torch.float32

# Classify a new website link
use = pd.read_excel('/Users/godspeed/Desktop/AEI_Paper_Yangdi/need_to_be_classified.xlsx')
# Print the dataset
print(use.head())
# for tuning process we store the link in a excel, while in the entile process use GPT_links
#website_links_cly = use['website_link'].tolist()
website_links_cly = GPT_links

classification_results = []
for new_website_link in website_links_cly:
    predi_class = classify_website(trained_model, tokenizer, new_website_link, device)
    classification_results.append(predi_class)
    print(f"The predicted class for the website '{new_website_link}' is {predi_class}")

use['predicted_class'] = classification_results
output_file_path = '/Users/godspeed/Desktop/AEI_Paper_Yangdi/already_classified_set.xlsx'
use.to_excel(output_file_path, index=False)