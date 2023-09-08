import pandas as pd
import argparse
import itertools
from tqdm import tqdm
import random
import time
import os
def remove_small_category(df, k, col):
    dict_category_to_count = df[col].value_counts().to_dict()
    small_category_list = []
    
    for key in dict_category_to_count.keys():
        if dict_category_to_count[key] < 10:
            small_category_list.append(key)
            
    df = df[df[col].isin(small_category_list) == False]

    return df

def sum_of_values(dictionary, keys):
    total_sum = 0


    for key in keys:
        if key in dictionary:
            total_sum += dictionary[key]
    return total_sum

def get_a_eligible_combination(dict_category_to_count, max_number_of_split, min_number_of_split, minimum_number_of_category):
    
    while True:
        all_keys = list(dict_category_to_count.keys())
        keys = random.sample(all_keys, minimum_number_of_category)
        sum_of_counts = sum_of_values(dict_category_to_count, keys)
        
        print('min: ' + str(min_number_of_split) + ' || max: ' + str(max_number_of_split) + ' || curr: ' + str(sum_of_counts))

        if sum_of_counts <=  max_number_of_split and sum_of_counts >= min_number_of_split:
            return keys
         
def get_df_for_seen_test(df, taxonomy_level):
    dict_taxonomy_level_to_count = df[taxonomy_level].value_counts().to_dict()
    
    train_split_list = []
    test_split_list = []
    
    for key in tqdm(list(dict_taxonomy_level_to_count.keys())):
        curr_category_df = df.loc[df[taxonomy_level] == key]
        num_for_test_split_of_curr_category = round(len(curr_category_df)*0.125)
        
        curr_test = curr_category_df.iloc[:num_for_test_split_of_curr_category]
        curr_train = curr_category_df.iloc[num_for_test_split_of_curr_category:]
        
        train_split_list.append(curr_train)
        test_split_list.append(curr_test)
    train_split = pd.concat(train_split_list, ignore_index=True)
    test_split = pd.concat(test_split_list, ignore_index=True)


    
    
    
    
    return train_split, test_split   

def main(args):
    df = pd.read_csv(os.path.join(args.input_dir, args.input_file_name) ,sep = '\t')
    df = df[df[args.taxonomy_level] != 'not_classified']
    df = df.sample(frac=1, random_state=42)
    
    df = remove_small_category(df, 10, args.taxonomy_level)
    original_size = len(df)
    print(df.columns)
    dict_family_to_count = df['family'].value_counts().to_dict()
    dict_taxonomy_level_to_count = df[args.taxonomy_level].value_counts().to_dict()
    
    sum_of_samples = len(df)
    
    max_number_of_sample = int(sum_of_samples * 0.105)
    min_number_of_sample = int(sum_of_samples * 0.095)
    min_number_of_category_for_unseen_hard = int(len(list(dict_family_to_count.keys()))*0.10)
    min_number_of_category_for_unseen_easy = int(len(list(dict_taxonomy_level_to_count.keys()))*0.10)
    
    
    # For hard unseen
    list_of_hard_unseen_split = get_a_eligible_combination(dict_family_to_count, max_number_of_sample, min_number_of_sample, min_number_of_category_for_unseen_hard)
    
    hard_unseen_df = df[df['family'].isin(list_of_hard_unseen_split)]
    df = df[~df['family'].isin(list_of_hard_unseen_split)]
    
    set1 = set(df['family'].unique())
    set2 = set(hard_unseen_df['family'].unique())
    print(set1.intersection(set2))
    
    
    
    # For easy unseen
    dict_taxonomy_level_to_count = df[args.taxonomy_level].value_counts().to_dict()
    list_of_easy_unseen_split = get_a_eligible_combination(dict_taxonomy_level_to_count, max_number_of_sample, min_number_of_sample, min_number_of_category_for_unseen_easy)
    easy_unseen_df = df[df[args.taxonomy_level].isin(list_of_easy_unseen_split)]
    df = df[~df[args.taxonomy_level].isin(list_of_easy_unseen_split)]

    et1 = set(df[args.taxonomy_level].unique())
    set2 = set(easy_unseen_df[args.taxonomy_level].unique())
    print(set1.intersection(set2))
    
    
    # For seen test
    seen_train, seen_test = get_df_for_seen_test(df, args.taxonomy_level)
    
    hard_unseen_df.to_csv(os.path.join(args.input_dir, args.taxonomy_level + '_hard_unseen_test.tsv'), sep='\t', index=False)
    easy_unseen_df.to_csv(os.path.join(args.input_dir, args.taxonomy_level + '_easy_unseen_test.tsv'), sep='\t', index=False)
    seen_test.to_csv(os.path.join(args.input_dir, args.taxonomy_level + '_seen_test.tsv'), sep='\t', index=False)
    seen_train.to_csv(os.path.join(args.input_dir, args.taxonomy_level + '_seen_train.tsv'), sep='\t', index=False)
    
    
    print('size of hard unseen test: ' + str(len(hard_unseen_df)))
    print('size of easy unseen test: ' + str(len(easy_unseen_df)))
    print('size of seen test: ' + str(len(seen_test)))
    print('size of seen train: ' + str(len(seen_train)))
    print(len(hard_unseen_df) + len(easy_unseen_df) + len(seen_test) + len(seen_train))
    print('size of all splits: ' + str(original_size))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy_level', type=str, default='species')
    parser.add_argument('--input_dir', type=str, default='data/bioscan_1m')
    parser.add_argument('--input_file_name', type=str, default='BioScan_Insect_Dataset_metadata_2.tsv')
    
    args = parser.parse_args()
    
    # random.seed(42)
    
    main(args)