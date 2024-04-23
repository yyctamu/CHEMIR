from datetime import datetime
import os
import sys
import numpy as np
import ssmpy

from algorithms import *
from cross_val import *
from data import *
from semsimcalculus import *
from myconfiguration import MyConfiguration as cfg
from dataset import upload_dataset

np.random.seed( 42 )

if __name__ == '__main__':

    start_time = datetime.now()

    # ----------------------------------------------------------------------------------------------------- #
    # check ontology database

    if os.path.isfile( cfg.getInstance().path_to_ontology ):
        print( "Database ontology file already exists" )

    else:
        print( "Database from owl does not exit. Creating..." )
        ssmpy.create_semantic_base( cfg.getInstance().path_to_owl,
                                    cfg.getInstance().path_to_ontology,
                                    "http://purl.obolibrary.org/obo/",
                                    "http://www.w3.org/2000/01/rdf-schema#subClassOf", "" )

    # ----------------------------------------------------------------------------------------------------- #
    # connect db
    #check_database()

    # ----------------------------------------------------------------------------------------------------- #
    # get the dataset in <user, item, rating> format

    ratings_original = upload_dataset( cfg.getInstance().dataset,
                                        cfg.getInstance().item_prefix )

    ratings2, original_item_id, original_user_id = id_to_index( ratings_original )  # are not unique

    # ratings = ratings2.drop(columns=["user", "item"])
    # ratings = ratings.rename(columns={"index_item": "item", "index_user": "user"})

    users_size = len( ratings2.index_user.unique() )
    items_size = len( ratings2.index_item.unique() )
    shuffle_users = get_shuffle_users( ratings2 )
    shuffle_items = get_shuffle_items( ratings2 )

    count_cv = 0

    # dictionary for saving the results of each cross validation for each model
    
    all_als = {}
    all_bpr = {}

    if cfg.getInstance().sim_metric in ('sim_lin', 'all'):
        all_onto_lin = {}

        all_als_onto_lin_m1 = {}
        all_als_onto_lin_m2 = {}

        all_bpr_onto_lin_m1 = {}
        all_bpr_onto_lin_m2 = {}
    if cfg.getInstance().sim_metric in ('sim_resnik', 'all'):
        all_onto_resnik = {}

        all_als_onto_resnik_m1 = {}
        all_als_onto_resnik_m2 = {}

        all_bpr_onto_resnik_m1 = {}
        all_bpr_onto_resnik_m2 = {}

    if cfg.getInstance().sim_metric in ('sim_jc', 'all'):
        all_onto_jc = {}

        all_als_onto_jc_m1 = {}
        all_als_onto_jc_m2 = {}

        all_bpr_onto_jc_m1 = {}
        all_bpr_onto_jc_m2 = {}

    cv_folds = cfg.getInstance().cv
    n = cfg.getInstance().n
    k = cfg.getInstance().topk

    for test_users in np.array_split( shuffle_users, cv_folds ):

        test_users_size = len( test_users )
        print( "number of test users: ", test_users_size )
        sys.stdout.flush()

    count_cv_items = 0
    for test_items in np.array_split( shuffle_items, cv_folds ):
            # models to be used

        test_items_size = len( test_items )
        print( "number of test items: ", test_items_size )
        sys.stdout.flush()
        print("------------------Debug Point 0 ------------------")
        ### prepare the data for implicit models
        ratings_test, ratings_train = prepare_train_test( ratings2, test_users, test_items )
        print("ratings test",ratings_test)
        print("ratings train",ratings_train)
        test_items = check_items_in_model( ratings_train.index_item.unique(), test_items )
        ratings_sparse = three_columns_matrix_to_csr( ratings_train )  # item, user, rating     ---- 102, 1152 item user sparse matrix with ratings
        print("test items",test_items)
        print("ratings test",ratings_test)
        print("------------------Debug Point 1 ------------------")
        print("test users", test_users)
        print("test users shape", test_users.shape)
        onto_lin, onto_resnik, onto_jc, als, \
        bpr, als_onto_lin_m1, als_onto_resnik_m1, \
        als_onto_jc_m1, bpr_onto_lin_m1, bpr_onto_resnik_m1, \
        bpr_onto_jc_m1, als_onto_lin_m2, als_onto_resnik_m2, \
        als_onto_jc_m2, bpr_onto_lin_m2, bpr_onto_resnik_m2, \
        bpr_onto_jc_m2 = get_evaluation(
            test_users, test_users_size, count_cv, count_cv_items, ratings_test,
            ratings_sparse, test_items, ratings2, original_item_id,
            cfg.getInstance().sim_metric )

        # add to dictionary
        all_als = add_dict( all_als, als, count_cv, count_cv_items )
        all_bpr = add_dict( all_bpr, bpr, count_cv, count_cv_items )

        if cfg.getInstance().sim_metric in ('sim_lin', 'all'):
            all_onto_lin = add_dict( all_onto_lin, onto_lin, count_cv, count_cv_items )

            all_als_onto_lin_m1 = add_dict( all_als_onto_lin_m1, als_onto_lin_m1, count_cv,
                                            count_cv_items )
            all_bpr_onto_lin_m1 = add_dict( all_bpr_onto_lin_m1,
                                            bpr_onto_lin_m1, count_cv,
                                            count_cv_items )
            all_als_onto_lin_m2 = add_dict( all_als_onto_lin_m2, als_onto_lin_m2, count_cv,
                                            count_cv_items )

        if cfg.getInstance().sim_metric in ('sim_resnik', 'all'):
            all_onto_resnik = add_dict( all_onto_resnik, onto_resnik,
                                        count_cv,
                                        count_cv_items )
            all_als_onto_resnik_m1 = add_dict( all_als_onto_resnik_m1,
                                                als_onto_resnik_m1, count_cv,
                                                count_cv_items )
            all_bpr_onto_resnik_m1 = add_dict( all_bpr_onto_resnik_m1,
                                                bpr_onto_resnik_m1, count_cv,
                                                count_cv_items )
            all_als_onto_resnik_m2 = add_dict( all_als_onto_resnik_m2,
                                                als_onto_resnik_m2, count_cv,
                                                count_cv_items )
            all_bpr_onto_resnik_m2 = add_dict( all_bpr_onto_resnik_m2,
                                                bpr_onto_resnik_m2, count_cv,
                                                count_cv_items )

        if cfg.getInstance().sim_metric in ('sim_jc', 'all'):
            all_onto_jc = add_dict( all_onto_jc, onto_jc, count_cv,
                                    count_cv_items )
            all_als_onto_jc_m1 = add_dict( all_als_onto_jc_m1,
                                            als_onto_jc_m1, count_cv,
                                            count_cv_items )
            all_bpr_onto_jc_m1 = add_dict( all_bpr_onto_jc_m1,
                                            bpr_onto_jc_m1, count_cv,
                                            count_cv_items )
            all_als_onto_jc_m2 = add_dict( all_als_onto_jc_m2, als_onto_jc_m2, count_cv,
                                            count_cv_items )
            all_bpr_onto_lin_m2 = add_dict( all_bpr_onto_lin_m2,
                                            bpr_onto_lin_m2, count_cv,
                                            count_cv_items )
            all_bpr_onto_jc_m2 = add_dict( all_bpr_onto_jc_m2, bpr_onto_jc_m2, count_cv,
                                            count_cv_items )

        sys.stdout.flush()
        count_cv_items += 1
        # TO REMOVE
        # if count_cv_items > 2:
        #    break

        count_cv += 1
        # TO REMOVE
        # if count_cv > 2:
        #        break
# ----------------------------------------------------------------------------------------------------- #
    # calculates mean and save to a csv file all metrics: [P, R, F, fpr, rr, nDCG, lauc] (preferencial order)
    path = '/mlData/results_nfolds'
    STR_N = '_nsimilar_'

    all_als = calculate_dictionary_mean( all_als, float( cv_folds * cv_folds ) )
    all_bpr = calculate_dictionary_mean( all_bpr, float( cv_folds * cv_folds ) )
    save_final_data( all_als, path + str( cv_folds ) + STR_N + str( n ) + "_ALS.csv" )
    save_final_data( all_bpr, path + str( cv_folds ) + STR_N + str( n ) + "_BPR.csv" )

    if cfg.getInstance().sim_metric in ('sim_lin', 'all'):
        all_onto_lin = calculate_dictionary_mean( all_onto_lin, float( cv_folds * cv_folds ) )
        all_als_onto_lin_m1 = calculate_dictionary_mean( all_als_onto_lin_m1, float( cv_folds * cv_folds ) )
        all_bpr_onto_lin_m1 = calculate_dictionary_mean( all_bpr_onto_lin_m1, float( cv_folds * cv_folds ) )
        all_als_onto_lin_m2 = calculate_dictionary_mean( all_als_onto_lin_m2, float( cv_folds * cv_folds ) )
        all_bpr_onto_lin_m2 = calculate_dictionary_mean( all_bpr_onto_lin_m2, float( cv_folds * cv_folds ) )
    
        path = create_directory('/mlData/sim_lin/') + 'results_nfolds'

        save_final_data( all_onto_lin, path + str( cv_folds ) + STR_N + str( n ) + "_onto_lin.csv" )
        save_final_data( all_als_onto_lin_m1, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_lin_m1.csv" )
        save_final_data( all_bpr_onto_lin_m1, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_lin_m1.csv" )
        save_final_data( all_als_onto_lin_m2, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_lin_m2.csv" )
        save_final_data( all_bpr_onto_lin_m2, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_lin_m2.csv" )

    if cfg.getInstance().sim_metric in ('sim_resnik', 'all'):
        all_onto_resnik = calculate_dictionary_mean( all_onto_resnik, float( cv_folds * cv_folds ) )
        all_als_onto_resnik_m1 = calculate_dictionary_mean( all_als_onto_resnik_m1, float( cv_folds * cv_folds ) )
        all_bpr_onto_resnik_m1 = calculate_dictionary_mean( all_bpr_onto_resnik_m1, float( cv_folds * cv_folds ) )
        all_als_onto_resnik_m2 = calculate_dictionary_mean( all_als_onto_resnik_m2, float( cv_folds * cv_folds ) )
        all_bpr_onto_resnik_m2 = calculate_dictionary_mean( all_bpr_onto_resnik_m2, float( cv_folds * cv_folds ) )
        
        path = create_directory('/mlData/sim_resnik/') + 'results_nfolds'
        
        save_final_data( all_onto_resnik, path + str( cv_folds ) + STR_N + str( n ) + "_onto_resnik.csv" )
        save_final_data( all_als_onto_resnik_m1, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_resnik_m1.csv" )
        save_final_data( all_bpr_onto_resnik_m1, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_resnik_m1.csv" )
        save_final_data( all_als_onto_resnik_m2, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_resnik_m2.csv" )
        save_final_data( all_bpr_onto_resnik_m2, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_resnik_m2.csv" )

    if cfg.getInstance().sim_metric in ('sim_jc', 'all'):
        all_onto_jc = calculate_dictionary_mean( all_onto_jc, float( cv_folds * cv_folds ) )
        all_als_onto_jc_m1 = calculate_dictionary_mean( all_als_onto_jc_m1, float( cv_folds * cv_folds ) )
        all_bpr_onto_jc_m1 = calculate_dictionary_mean( all_bpr_onto_jc_m1, float( cv_folds * cv_folds ) )
        all_als_onto_jc_m2 = calculate_dictionary_mean( all_als_onto_jc_m2, float( cv_folds * cv_folds ) )
        all_bpr_onto_jc_m2 = calculate_dictionary_mean( all_bpr_onto_jc_m2, float( cv_folds * cv_folds ) )
        
        path = create_directory('/mlData/sim_jc/') + 'results_nfolds'

        save_final_data( all_onto_jc, path + str( cv_folds ) + STR_N + str( n ) + "_onto_jc.csv" )
        save_final_data( all_als_onto_jc_m1, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_jc_m1.csv" )
        save_final_data( all_bpr_onto_jc_m1, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_jc_m1.csv" )
        save_final_data( all_als_onto_jc_m2, path + str( cv_folds ) + STR_N + str( n ) + "_als_onto_jc_m2.csv" )
        save_final_data( all_bpr_onto_jc_m2, path + str( cv_folds ) + STR_N + str( n ) + "_bpr_onto_jc_m2.csv" )

    # ----------------------------------------------------------------------------------------------------- #
    # save time process
    end_time = datetime.now()
    with open( '../info_process.txt', 'a' ) as f:
        f.write( "Date: {date} \nDuration: {duration}\n".
            format(
            date=datetime.now(),
            duration=end_time - start_time )
        )
        f.write( "Database: {db} \nDataset: {ds} \nOntology: {onto}\n\n".
            format(
            db=cfg.getInstance().database,
            ds=cfg.getInstance().dataset,
            onto=cfg.getInstance().item_prefix )
        )
        f.close()
