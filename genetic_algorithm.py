#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:13:11 2019

@author: krisjan

genetic algorithm that generates a population of lists of floats and ints and
then breeds and mutates the list
"""
# import modules
import pandas as pd
import numpy as np
import random
#import pbar
from math import floor, ceil


# generate population
def gen_pop(list_of_types, lower_bounds, upper_bounds, size, ind_seed_dataframe=None):
    if ind_seed_dataframe is None: 
        len_seed = 0
    else:
        len_seed = len(ind_seed_dataframe)
    no_random_inds = size - len_seed
    df_inds = []
    for dtype, lb, ub in zip(list_of_types, lower_bounds, upper_bounds):
        if dtype == 'int':
            rand_list = list(np.random.randint(lb,ub,no_random_inds))
        else:
            rand_list = list(np.random.uniform(lb,ub,no_random_inds))
        df_inds.append(rand_list)
    df_inds = pd.DataFrame(df_inds).T
    if ind_seed_dataframe is not None:
        df_inds = pd.concat([ind_seed_dataframe,df_inds],ignore_index=True)
    return df_inds

# breed
def breed(df_inds, df_score, nr_children_limit):
    list_of_pairs = [(df_inds.index[p1], df_inds.index[p2]) 
                                            for p1 in range(len(df_inds)) 
                                            for p2 in range(p1+1,len(df_inds))]
    list_of_probs =  pd.Series([(df_score[p1]+df_score[p2])/2 
                                            for p1 in range(len(df_score)) 
                                            for p2 in range(p1+1,len(df_score))])
    list_of_probs.sort_values(ascending=False, inplace=True)
    breed_pair_index = list(list_of_probs.index)
    breed_limit = min([len(breed_pair_index)*2,nr_children_limit])
    breed_frame = pd.DataFrame()
    # GAAN HIER AAN
    for i in breed_pair_index[:breed_limit]:
        index_1, index_2 = list_of_pairs[i]
        str_len = len(df_inds.columns)-1
        new_ind_1 = pd.DataFrame()
        new_ind_2 = pd.DataFrame()
        split_point = random.choice(range(str_len))
        new_ind_1 = pd.concat([df_inds.loc[index_1].iloc[:split_point],
                               df_inds.loc[index_2].iloc[split_point:]])
        new_ind_2 = pd.concat([df_inds.loc[index_2].iloc[:split_point],
                               df_inds.loc[index_1].iloc[split_point:]])
        name_1 = str(index_1) + '_' + str(index_2)
        name_2 = str(index_2) + '_' + str(index_1)
        breed_frame[name_1] = new_ind_1
        breed_frame[name_2] = new_ind_2
    return breed_frame.T

# mutate
def mutate(df_survivors, list_of_types, lower_bounds, upper_bounds, probability = .01, 
           strength = .2):
    mutation_probs = np.random.uniform(0,1,(len(df_survivors),len(df_survivors.columns)))
#    print(mutation_probs)
    mutation_multiplier = (mutation_probs <= probability).astype(int)
#    print(mutation_multiplier)
    mutations = []
    for dtype, lb, ub in zip(list_of_types, lower_bounds, upper_bounds):
        if dtype == 'int':
            rand_list = list(np.random.randint(lb,ub,len(df_survivors)))
        else:
            rand_list = list(np.random.uniform(lb,ub,len(df_survivors)))
        mutations.append(rand_list)
    mutations = mutation_multiplier * np.array(mutations).T * strength
#    print(mutations)
    df_mutated = df_survivors - df_survivors * mutation_multiplier + \
                    (mutation_multiplier * df_survivors * (1 - strength) + mutations)/2
    return df_mutated
            
def evolve(list_of_types, lower_bounds, upper_bounds, pop_size, ind_seed_dataframe,
           generations, eval_func, mutation_prob=.01, mutation_str=.2, perc_strangers=.05, 
           perc_elites=.1, old_scores=None, save_gens=False):
    nr_strangers = ceil(perc_strangers * pop_size)
    nr_elites = ceil(perc_elites * pop_size)
    nr_bred = floor((pop_size - nr_strangers - nr_elites)/2)
    
    if old_scores is None:
        print('-------generating population--------')
        
        df_inds = gen_pop(list_of_types, lower_bounds, upper_bounds, pop_size, ind_seed_dataframe)
        
        df_score = df_inds.apply(eval_func, axis=1)
    else:
        df_inds = ind_seed_dataframe
        df_score = old_scores
    
    print('best individual',list(df_inds.loc[df_score.idxmax()]))
    print('best score:',df_score.max())
    
    for gen_num in range(1,generations):
        print('-------generation',gen_num)
        df_elite_score = df_score.sort_values(ascending=False).iloc[:nr_elites]
        df_elites = df_inds.loc[df_elite_score.index]
        
        df_new_gen = breed(df_inds, df_score, nr_bred)
        df_new_gen = mutate(df_new_gen, list_of_types, lower_bounds, upper_bounds,
                            mutation_prob, mutation_str)
        
        #prevent duplication of individuals
        df_inds = pd.concat([df_elites,df_new_gen]).drop_duplicates()
        df_new_gen_score = df_inds.iloc[nr_elites:].apply(eval_func, axis=1)
        
        nr_str_fill = max([pop_size-len(df_inds),nr_strangers])
        df_strangers = gen_pop(list_of_types, lower_bounds, upper_bounds, nr_str_fill)
        df_strangers_score = df_strangers.apply(eval_func, axis=1)
        
        df_inds = pd.concat([df_inds, df_strangers],ignore_index=True)
        df_score = pd.concat([df_elite_score,df_new_gen_score,df_strangers_score], ignore_index=True)
        print('best individual',list(df_inds.loc[df_score.idxmax()]))
        print('best score:',df_score.max())
        if save_gens:
            df_inds.to_csv('generation_'+str(gen_num)+'.csv')
            df_score.to_csv('scores_'+str(gen_num)+'.csv')
    return df_inds, df_score