### Note

This repository and code is made for NASA’s Risky Space Challenge by [Axel Ország-Krisz Dr.](https://github.com/okaxel) and [Richárd Ádám Vécsey Dr.](https://github.com/richardvecsey). **Hyperrixel** has a double meaning. First of all it is a *fantasy name for us* and it is our joint github account where we can develop equally. We develop together to solve real-life problems and not for rocketing one or another of our personal github accounts.

---

# What is Risky Space about?

It is a purely NLP-based risk identification and prediction method to reduce the chance of occurring unknown risks during a project’s lifetime. During the challenge we made a proof of concept to demonstrate how we can handle risks. 


# Problem statement

Extracting risks from text documents is a hard task even for humans. There are a lot of causes behind this problem.
1. The author of the document or report knows nothing about the potential risk and their consciousness does not cover it. In this case there will be no evidence in the document about the concrete risk.
2. The author of the document has some vision about the risk, but this remains on the unconscious level. In this case the document can contain words that refer to the risk.
3. The author knows the risk and mentions it in the document without any circumscription. The document contains the name or attribute of risks but nothing more. 
4. The author knows the risk very well and wants to hide it from the document intentionally to mislead the reader or hide responsibilities. When the text is processed with a static NLP-based AI/ML code, authors can learn how to deceive it. Creating documents from time to time is an evolving process. Extracting risk from a document like this can be possible, if the algorithms evolve from time to time. It is like a well-known cop-robber problem: a smarter robber results in smarter cops (better criminal investigation methods), while smarter cops result in smarter robbers. 
5. The author knows the risk very well and describes it explicitly in the document. This is the easiest challenge from all of the possible cases.

Between AI/ML developers, the importance of datasets always pops out. This is no different in this special challenge. There are no available rich and well-structured data sources. In most cases, the available datasets are really unbalanced with a lot of negative samples. On the other hand, the variability of positive cases is really huge.

Feeding a model with various length data inputs is hard. Technologically, it is possible to handle the differ inputs with special embedding layers that have an infinite or nearly infinite amount of input nodes and a specified amount of output nodes. However, the data and information extraction process is really different when we focus on a sentence based on 10 words or a 100 pages pure text document. 

# What are the elements of this method?

## risks 

![Figure 01](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2001%20-%20risks.png "risks")


### risk categories

- Financial (Cost) F
- Natural N
- Programmatic P
- Resource R
- Safety Sa
- Schedule Sh
- Scientific Sc
- Social So
- Technical T

These categories rhyme with GCD’s existing risk and affinity categories. However, we add more depth to the system by using more categories. 

### risks as output of the models

Since GCD project managers already categorize risk in multiple ways, we decided to keep this method and make predictions for both ways: 5x5 risk severity-likelihood matrix and an extended list of risk categories. Using more risk categories helps to distinguish risks that had been identified as unknown risks. Furthermore, one of our models highlights hidden risks from the input texts to reduce the chance of occurring unexpected risks during the project lifetime.


## bag-of-words

![Figure 06](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2006%20-%20datasets.png "datasets")


### filtered words

We collected the most common first or last names in the US and global geolocation names with time zone data. After the lemmatization and removal of duplicates, **719,797 unique records** remain. Filtered words dataset helps to clean the input data from unnecessary noise, since names can lead to bias. For example A, B and C people usually work on projects with higher risks than D and E people. When A and C people get a new project with lower risk, a machine learning algorithm can classify falsely higher the risk due to the occurrences of their names in the project description. The geolocation names are similar. 


### known words

The known words dataset consists of *5 different parts*: the most common 10k English word lemmas derived from the most common 300k English words, more than 5k science related word lemmas, ASCII symbols, digits and one token for other words. All of them build up a known word dataset with **15,263 unique tokens**.


## models

![Figure 04](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2004%20-%20models.png "models")


### model 1

Model 1 identifies the risk category from our nine-item list.


### model 2

Model 2 predicts severity and likelihood factor for each risk category. The prediction is a single value for each factor in each category. The value is between 0.0 and 1.0, where 0.0 means zero severity or likelihood, and 1.0 means maximum severity or likelihood.


### model 3

Model 3 has a single binary output. Its predictions show whether a document contains potential risk or not. This model is to recognize unexpected, unknown or unclassified risks.


# How does it work?

## inference

![Figure 02](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2002%20-%20inference%20workflow.png "inference")


## training

![Figure 03](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2003%20-%20training%20workflow.png "training")


## iterative text analysis

![Figure 05](https://github.com/hyperrixel/RiskySpace/blob/main/figs/Figure%2005%20-%20iterative%20text%20analysis.png "iterative text analysis")


# How to use


## install

```
git clone https://github.com/hyperrixel/RiskySpace.git
```

```
pip install -r requirements.txt
```

## run


### inference

Inference can be done by command line or programmatically.


#### inference from command line

Command line inference is simply with inference.py from the root directory.

```
inference.py –file FILE_PATH
```

or

```
inference.py –text “TEXT”
```


#### programmatically inference

Inference can be accessed programmatically through the functions of the inference submodule.


### training

Training is available through a factory training loop with the simple call of train.py from the root directory.


# Data sources:

*Annual reports*
- Annual reports from Lockheed Martin between 1995 and 2020
- https://www.lockheedmartin.com/en-us/news/annual-reports.html
- Annual reports from Virgin Galactic between 2019 and 2020
- https://investors.virgingalactic.com/financials/Annual-Reports-and-Proxy-Forms/default.aspx
- Annual reports from Coca-Cola between 2004 and 2021
- https://investors.coca-colacompany.com/filings-reports/annual-filings-10-k

*Books from Project Gutenberg*
- The Chemistry of Food and Nutrition by A. W. Duncan
- https://www.gutenberg.org/ebooks/15237
- Evolution in Modern Thought by William Bateson et al.
- https://www.gutenberg.org/ebooks/22430
- Yeast by Thomas Henry Huxley
- https://www.gutenberg.org/ebooks/2938
- Curiosities of the Sky by Garrett Putman Serviss
- https://www.gutenberg.org/ebooks/6630
- Relativity: The Special and General Theory by Albert Einstein
- https://www.gutenberg.org/ebooks/30155
- The Murder on the Links by Agatha Christie
- https://www.gutenberg.org/ebooks/58866
- The Picture of Dorian Gray by Oscar Wilde
- https://www.gutenberg.org/ebooks/4078
- Alice in Wonderland by Lewis Carroll and Alice Gerstenberg
- https://www.gutenberg.org/ebooks/35688

*Filtered words*
- first names
- https://www.ssa.gov/oact/babynames/limits.html

- last names
- https://www.census.gov/topics/population/genealogy/data/2000_surnames.html
- geolocation names
- https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table/?disjunctive.cou_name_en&sort=name

*Known words*
- scientific words, part 1
- https://data.europa.eu/data/datasets/euroscivoc-the-european-science-vocabulary?locale=en
- scientific words, part 2
- https://wordnet.princeton.edu/download
- most common English words
- http://norvig.com/ngrams/


# Future plans

First of all we want to speed up our algorithms. Secondly, we want to create a better, more balanced dataset for training. Training a model with small datasets leads to overfitting. For building a more robust model there is a need for more data. It would be good to have the chance to work with more data from GCD. We can translate the algorithms into C++, Java  or any other languages. Lastly, this whole process can be part of a research, since it hides great opportunities. We are not typical academic people, but we would be happy to cooperate with researchers, scientists or NASA’s staff to help them with our findings.

# License

This repository is under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) permissive license. For more information see [LICENSE.md](https://github.com/hyperrixel/RiskySpace/blob/main/LICENSE)
