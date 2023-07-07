import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)
# Load the dataset and see how many samples we actually downloaded
dataset = pd.read_csv("data/full_dataset.tsv", sep='\t', usecols=range(1,11))
dataset.dropna(subset=["nucleotides"], inplace=True)
dataset['nucleotides'] = dataset['nucleotides'].str.replace('[^ACGTN]', 'N', regex=True)
#df.fillna('', inplace=True)
print(dataset.groupby('phylum_name').nunique().to_markdown())
dataset['sequence_len'] = dataset['nucleotides'].apply(lambda x: len(x))
dataset['sequence_len'].hist(bins=100)
plt.title('Length distribution original dataset')
plt.savefig('original_length_distribution.png', dpi=150)

#print(df.to_markdown())



## Finding Corrupted Species
study =  dataset.duplicated(subset=['nucleotides'])

study = dataset.loc[study][['nucleotides', 'processid', 'species_name']]
study = study.drop_duplicates(subset='species_name')


corrupted_species = []
corrupted_ids = []


c = 0
for group, nuc_df in study.groupby(['nucleotides']):
   if len(nuc_df) > 1:
       corrupted_species.extend(list(nuc_df.species_name))
       corrupted_ids.extend(list(nuc_df.processid))

print(f'There are {len(corrupted_species)} species that share the same sequence with other species')
#print(corrupted_ids)


#Building the Datasets 

dataset['nucleotides'] = dataset['nucleotides'].apply(lambda x: x.rstrip('N'))

dataset = dataset [dataset ['nucleotides'].apply(lambda x: len(x)>200)
                & dataset ['nucleotides'].apply(lambda x: x.count('N') < len(x)*0.5)]



dataset.drop_duplicates(subset=['nucleotides'], inplace=True)
print(dataset.groupby('phylum_name').nunique().to_markdown())

dataset['sequence_len'].hist(bins=100)
plt.figure()
plt.title('Length distribution after removing duplicates and trailing Ns')
plt.savefig('new_length_distribution.png', dpi=150)

target_level = 'species_name'

# Select the classes [genera or species] with more than 
# 50 samples per class and random sampling of 50 sequences
# per each overrepresented class

filtered = dataset[(dataset[target_level].notna()) & ~(dataset[target_level].isin(corrupted_species))]
s = filtered.groupby(target_level).sampleid.count()
l = s[s > 50].index.to_list()
print(f'There are {len(l)} species with more than 50 samples per species')
filtered = filtered[filtered[target_level].isin(l)]

#Sample 100 species at random to be unseen by our model for validation.
selected_species = random.sample(l, k=100)

#print(len(set(selected_species)))
unseen = filtered[filtered[target_level].isin(selected_species)]

train = pd.concat([filtered, unseen, unseen]).drop_duplicates(keep=False)



train  = train.groupby(target_level).sample(n=50, random_state=1)
unseen = unseen.groupby(target_level).sample(n=50, random_state=1)

#print(train)
train.to_csv('data/supervised.tsv', sep='\t', index=False)

#print(unseen)
unseen.to_csv('data/useen.tsv', sep='\t', index=False)

pre_training = pd.concat([dataset, train, train, unseen, unseen]).drop_duplicates(keep=False)
#print(pre_training)
pre_training.to_csv('data/pre_training.tsv', sep='\t', index=False)
