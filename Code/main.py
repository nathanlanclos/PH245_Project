#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 8 19:36:20 2025

@author: jonny
"""
import pickle


from features import ESMFeatureExtractor, BlosumFeatureExtractor, SequenceFeatureExtractor, StructureFeatureExtractor, StructureGenerator, ESM3FeatureExtractor
from models import SolubilityPredictor
if __name__ == "__main__":
    '''
    input_csv_paths = ["dataset/eSol_test.csv", 'dataset/eSol_train.csv']
    output_paths = ['dataset/processed/eSol_test.csv', 'dataset/processed/eSol_train.csv']
    pdb_dir ='dataset/structures/eSol_full/'
    
    for i in range(2):
        df = StructureGenerator().generate_structures_from_csv(input_csv_paths[i], pdb_dir)
        df = StructureFeatureExtractor.generate_structure_features(input_csv_paths[i], pdb_dir, output_path=output_paths[i])
        df = SequenceFeatureExtractor().generate_sequence_features(df, output_path=output_paths[i])
        df = BlosumFeatureExtractor().generate_blosum_embeddings(df, output_path=output_paths[i])
        df = ESMFeatureExtractor().generate_ESM_embeddings(df, output_path=output_paths[i]) 
    
    df = StructureGenerator().generate_structures_from_csv("dataset\S.cerevisiae_test.csv", "dataset\structures\S.cerevisiae")
    df = StructureFeatureExtractor.generate_structure_features("dataset\S.cerevisiae_test.csv", "dataset\structures\S.cerevisiae", output_path="dataset\processed\S.cerevisiae_test.csv")
    df = SequenceFeatureExtractor().generate_sequence_features(df, output_path="dataset\processed\S.cerevisiae_test.csv")
    df = BlosumFeatureExtractor().generate_blosum_embeddings(df, output_path="dataset\processed\S.cerevisiae_test.csv")
    df = ESMFeatureExtractor().generate_ESM_embeddings(df, output_path="dataset\processed\S.cerevisiae_test.csv") 
    '''
    predictor = SolubilityPredictor(k_folds=3)
    predictor.split_data('dataset/processed/eSol_train3.csv', "dataset\processed\eSol_test3.csv")
    # Define the optimal parameters for the model
    best_params = {'n_estimators': 200, 
                'learning_rate': 0.034644213887322986, 
                'max_depth': 6, 
                'min_child_weight': 9.71406913873956, 
                'gamma': 0.14595452117738994, 
                'subsample': 0.7745464078642961, 
                'colsample_bytree': 0.5504898392099907, 
                'reg_alpha': 1.5166161603842931, 
                'reg_lambda': 1.0900962363439852}
    
    xgb_reg = predictor.cross_validate_xgboost(n_optuna_trials=20)

    '''
    with open('Code/models/xgb_reg.pkl', 'wb') as f:
        pickle.dump(xgb_reg, f)
    '''
