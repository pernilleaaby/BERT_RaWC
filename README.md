# BERT Representation Analysis
This project is analysing BERT word representations. 
We will provide a brief description of each script and in which order the script has been run. 

## Distilled BERT EMbeddings
### corput_processing/find_top_dictionary.py
The script tokenize the whole Norwegian News Corpus dataset and counts the frequency of each word. This way we can find the most frequent words in our corpus and use as our word embedding vocabulary. We also cross-check with a Norwegian word list, and only include the word if it can be found in the word list.  
### corpus_processing/make_news_subset.py
This script uses the top dictionary to reduce the News Corpus to a smaller part, in which we know we can find a sufficient number of contexts for each of the words in the top dicitonary. We do this simply to reduce the processing time when we find our word embeddings through BERT. 
### embedding_generation/generate_averaged_embeddings.py
This script use the reduced News Corpus to produce our word embedding vocabulary for Norwegian. It inferences the text through BERT to obtain the contextual embeddings for each word and then take the average of all the contextual embeddings for each word. We keep a "running" average dictionary for the word embeddings to deal with memory problem (too many embeddings in RAM at one time consumes the RAM fast). It took around 24 hours to run the script on a GPU with ~14 GB RAM (the time of course highly depend on the resource). 

## Cross-Lingual Analysis
### embedding_generation/generate_averaged_embeddings_en.py
This script is very similar to the script that generates the Norwegian word embeddings, there is just some details about the dataset and vocabulary that are switched out. We only use the Brown corpus for the English text Corpus, which is significantly smaller than the Norwegian News corpus, so this script runs much faster. 

### embedding_analysis/en2no_MT_static.py
The script evaluates the word retrieval for the static word embeddings from English source to Norwegian target. 
### embedding_analysis/plot_static_word_retrieval.ipynb
The script plots the results from the static word retrieval experiment. 

### embedding_analysis/cross_lingual_word_disambiguation.py
The script tests a few qualitative examples for word retrieval from contextual embedding as source to static embedding as target. We do this to understand better how a word can find its relevant meaning in the given language and relate to similar words in both English and Norwegian. 

### embedding_analysis/mt_tsne_word_plot.ipynb
The script takes a set of 5 words, in both English and Norwegian, and finds the word pairs in a small sample of paralell sentences. This way we can visually show how the words relate in a scatter plot. 
### embedding_analysis/contextual_word_retrieval.py
The script evaluates contextual word retrieval from English to Norwegian with parallel sentences and aligned word pairs. 
### embedding_analysis/plot_contextual_word_retrieval.ipynb
The script plots the results from the contextual word retrieval. 

### embedding_analysis/language_detection.ipynb
The script detects the language of each word in the static word embedding vocabularies for both English and Norwegian. 
### embedding_analysis/qualitative_language_detection_analysis.ipynb
The script evaluates the language detection method through qualitative sentence examples which include both English and Norwegian words mixed together. 

## Real, Wrong, ISO Analysis
### corpus_processing/generate_confusion_candidates.py
The script uses the top dictionary from the Norwegian News corpus to generate a confusion set for each word. The confusion set consists of similar words based on wither phonetic letter distance or normal letter distance. 
### corpus_processing/norne_add_confusion.py
The script generates wrong context by switching out a word with one of the confusion candidates in its confusion set. We use the text passages from the NorNe dataset as our testing corpus. 

### embedding_analysis/static_dynamic_comparison.py
The script evaluates the number of matches from the contextual word embeddings from the NorNe text passages into the static word embedding vocabulary. 
### embedding_analysis/plot_static_dynamic_results.ipynb
The script plots the results from the real versus wrong context, highest match experiment. 

### real_wrong_iso_comparison.ipynb
The script compares a word collection from real, wrong and isolated context through visual plots and distributions. 


## utils folder
### load_dict.py
Helper functions for loading the word lists for Norwegian. 
### model_utils.py
Helper functions for BERT processing functions. The functions that generates word embeddings. 
### levensthein_utils.py
Helper funcitons for finding letter distances. We use Damerau Levensthein distance. 
