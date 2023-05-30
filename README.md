# GPTdvHDSKG
# This is a project about KG

## Current status of this project:

1. We have prepared a GPT-API based website links collector, which aims to use GPT's inner trained data to efficiently provide us with links about the key domain words (hypertension, hypercholesterol, diabetes, etc.) 
2. A webclassifier is prepared. This classifier is trained by leveraging pre-trained DistilBert (it can be locally deployed on a macbookpro with 16 core GPU). And the features chosen are fully content based. 

## About Code file:
for each code.py files, I already commit the basic funcitions achieved by each files. And inside the file, for each code parts, I add description of its function.


## Running flow of codes:

Actually, for functions mentioned above have been wrappered into a file named GPT_&_Classifier.py. I will briefly describe the running flow of it:
1. the GPT-API will be used to collect the links about a keywords embedded into the promt text(you can change it). 
2. for begginner of this project, you can use the prepared training_links_set.xlsx to scrape the webpage contents and clean the content with defined pre-process code, Or you can directly use the training_text_set.xlsx to training the Bert classifier.
3. after classifier prepared, you can use the classifier to classify the links collected by GPT-API. it will be divided into two class(1,0), 1 refers to key words related, 0 refers to meaningless websites. 
