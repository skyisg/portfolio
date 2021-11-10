#!/usr/bin/env python
# coding: utf-8

import sys, os

pardir_path = '/'.join(os.getcwd().split('/')[:-1])

sys.path.append(pardir_path)

# public lib
import pandas as pd

# private lib
from classification.news_classification import Text_Preprocessing

# Text_Preprocessing class import 
tp = Text_Preprocessing()

# define saving function
def get_excel_saving_df(cat_list):
    cat_df = pd.DataFrame({'original_category':cat_list, 'renamed_category':0})
    return cat_df


# get category excel file for  mind dataset

#load dataset
mind_dir_path = 'data/MIND/'

mind_train_path = mind_dir_path + 'train/news.tsv'

train_df = tp.parse_mind_dataset(mind_train_path)


# mind_cats_df
mind_categories = train_df.Category.value_counts().index
mind_subcategories = train_df.SubCategory.value_counts().index

mind_cat_df = get_excel_saving_df(mind_categories)
mind_subcat_df = get_excel_saving_df(mind_subcategories)


# get category excel file for huffpost dataset

huff_dir_path = 'data/HUFFPOST/'
huff_file_name = os.listdir(huff_dir_path)[0]
huff_file_path = huff_dir_path + huff_file_name

huff_df = tp.parse_huff_dataset(huff_file_path)

huff_cat = [index.lower() for index in huff_df.category.value_counts().index]

huff_cat_df = get_excel_saving_df(huff_cat)


# get category excel file for ag dataset

ag_cat = ['world', 'sports', 'business', 'sci/tech']

ag_cat_df = get_excel_saving_df(ag_cat)


# get category excel file for webapp dataset

webapp_cat = ['world', 'entertainment', 'sports', 'technology', 'politics', 'science', 'automobile']

webapp_cat_df = get_excel_saving_df(webapp_cat)


# save categories in excel
excel_save_path = 'data/merge/raw_data/original_category.xlsx'

with pd.ExcelWriter(excel_save_path) as writer:
    mind_cat_df.to_excel(writer, sheet_name='MIND')
    mind_subcat_df.to_excel(writer, sheet_name='MIND_sub')
    huff_cat_df.to_excel(writer, sheet_name='HUFFPOST')
    ag_cat_df.to_excel(writer, sheet_name='AG')
    webapp_cat_df.to_excel(writer, sheet_name='WEBAPP')
    
print('saved')

