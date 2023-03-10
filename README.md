# Bumblebee

A pre-trained representation from a transformers model for inference on insect DNA barcoding data. 

*Note*: If you have been here before, you will note that the code is not a mess anymore. Shoutout to Niousha and Monireh. We do not need Hugging-Face as the entire architecture was implemented from scratch by them


1. Make sure you have all the required libraries before running (remove the --no-index flags if you re not training on CC)

```
./DNABERT.sh
```

Our new vocabulary consists of {A,C,G,T,-,N}.

**To-Do**:
Muli-GPU support
