# word-embeddings
<h1>README</h1>

```extractData.py``` file unzips the dataset and converts the move reviews (json) into a txt file ```reviews.txt``` which is further used for preprocessing. 

```preProcessing.py``` creates a dictionary and on this basis replace the less frequency words with \<UNK\> and saves this file in preProcess.txt which is used for model training. 


<h3>SVD</h3>	

1. `SVD` generates co-occurrence matrix and then SVD matrix. The final embedding  are saved in a file. 

   [Embeddings.txt](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/abhishek_shar_students_iiit_ac_in/ES-xuABRSZpEkpTUUjYSmAgBNWHy9iM4XykLa-RbCu3RbQ?e=5VhzSA)

2. Using this embedding , `generateResult.py` file loads the embedding layers . Running  ```python3 generateResult.py``` on console, give a word as input and top 10 words printed as a result. 

   ```
   Enter the word: funny
   25089it [00:00, 32070.94it/s]
   Word:  funny
   Top 10 similar words:  ['scary', 'entertaining', 'sad', 'hilarious', 'sexy', 'clever', 'cool', 'suspensful', 'touching', 'cute']
   ```

   

<h3>CBOW</h3>

1. `CBOW` class generates embedding using CBOWmodel class saved in the below file. 

   [Embeddings.txt](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/abhishek_shar_students_iiit_ac_in/EU4_K_uNN8FMv_R2y6gDszoB1kAf9O5qyW2gEzPCSgAUNA?e=rKNdxS)

2. Downloading the embedding in the same directory, run ```python3 generateResult.py``` and same format appears for this also. 

   ```
   Enter the word: funny
   10it [00:00, 104596.11it/s]
   Word:  funny
   Top 10 similar words:  ['hilarious', 'cute', 'scary', 'freaky', 'smart', 'goofy', 'dumb', 'unintentionally', 'touching', 'comical']
   ```

   
