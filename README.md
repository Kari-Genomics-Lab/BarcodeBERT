# Bumblebee
A pre-trained representation from a transformers model for inference on insect DNA barcoding data. 


1. Make sure you have all the required libraries before running (remove the --no-index flags if you re not training on CC)

```
./DNABERT.sh
```

2. The notebooks are in the folder examples
3. You'll have to change the config files before running locally, all the PATHS are fixed to match my CC cluster
4. If your're testing the pre-trained models, download them directly from DNABERT and save them in ft/ 
4. When running the Pre-Training script, make sure that your data is tokenized, you'll have run a separate script for that but don't worry if you run on our dataset.
5. Our new vocabulary consists of {A,C,G,T,-,N}.


*Disclaimer*: The code is still a mess, I am still trying to figure it out.

**To-Do**:
Jupyter Notebook for Pre-Trainig [Next Week]
Implement the MAP for evaluation [Next Week]
