# Clickbait Detector
## Definition
Clickbait detector is a classifier for clickbait headlines. This repository presents two approaches: naïve word distribution and naïve Bayes classifier.
## Instructions
Change to the desired classifier directory. Run  
```
python3 train.py
``` 
To use such classifier, run
```
python3 guess.py
``` 
In the wordcount classifier, change the address of the server to your Redis server address.
## Dependencies
Required programs:
- Python 3
- Redis

Python dependencies:
- nltk (all-nltk must be downloaded from the nltk download interface)
- re 
- redis

## Credits
Authors:
- Miguel Miranda (@mmiranda96)
- Rosa Ramírez (@rosamariaramirez)

Titles corpus are from [saurabhmathur96's clickbait detector] (https://github.com/saurabhmathur96/clickbait-detector).