import ssmpy
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import ssmpy
import time
from database import *

pd.set_option( 'display.max_columns', None )


def calculate_sim_it1_it2(it1, it2, e1, name_prefix):
    """
    Calculate similarity between it1 and it2 and insert in database
    :param it1: entity 1 (id)
    :param it2: entity 2 (id)
    :param e1: ontology (id)
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return
    """

    if check_if_pair_exist( it1, it2 ) is False:

        if name_prefix == 'HP_':
            it2 = '{:07d}'.format( it2 )
        it2_ = name_prefix + str( it2 )
        # print("   ", it2_)
        try:
            start = time.time()
            e2 = ssmpy.get_id( it2_ )
            items_sim_resnik = ssmpy.ssm_resnik( e1, e2 )
            items_sim_lin = ssmpy.ssm_lin( e1, e2 )
            items_sim_jc = ssmpy.ssm_jiang_conrath( e1, e2 )
            end = time.time()
            print( "unique calc: ", end - start )
            insert_row( it1.item().astype( int ), it2.item().astype( int ), items_sim_resnik, items_sim_lin, \
                items_sim_jc )
            sys.stdout.flush()
        except TypeError:
            print( it1, " or ", it2, " not found." )
    else:
        print( it1, it2, " pair already exists" )

# ----------------------------------------------------------------------------------------------------- #

# @jit(target="cuda")
def calculate_sim_it1_it2_test_gpu(it1, e1, onto_ids, name_prefix):
    """
    Calculate similarity between it1 and it2 and insert in database
    :param it1: entity 1 (id)
    :param e1: id of the entity 1
    :param onto_ids:
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return: 
    """
    for it2 in onto_ids:
        if not check_if_pair_exist( it1, it2 ):

            if name_prefix == 'HP_':
                it2 = '{:07d}'.format( it2 )

            it2_ = name_prefix + str( it2 )

            try:
                e2 = ssmpy.get_id( it2_ )
                items_sim_resnik = ssmpy.ssm_resnik( e1, e2 )
                items_sim_lin = ssmpy.ssm_lin( e1, e2 )
                items_sim_jc = ssmpy.ssm_jiang_conrath( e1, e2 )
                insert_row( it1.item().astype(int), it2.item().astype(int), items_sim_resnik, items_sim_lin, items_sim_jc )

                sys.stdout.flush()
            except TypeError:
                print( it1, " or ", it2, " not found." )

        else:
            print( it1, it2, " pair already exists" )
            continue

# ----------------------------------------------------------------------------------------------------- #

def calculate_semantic_similarity(onto_ids: object, name_prefix: object, train: object) -> object:
    """
    Calculate semantic similarity 
    :param onto_ids: list of ids (int)
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :param train: list of train dataset
    :return:
    """

    conf = cfg.getInstance()

    ssmpy.semantic_base( conf.path_to_ontology )

    print( "test size: ", len( onto_ids ) )
    print( "train size: ", len( train ) )

    count = 0
    for it1 in onto_ids:
        if name_prefix == 'HP_':
            it1_ = '{:07d}'.format( it1 )
            it1_ = name_prefix + str( it1 )
        else:
            it1_ = name_prefix + str( it1 ) 
        print( it1_ )
        e1 = ssmpy.get_id( it1_ )
        # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool( 30 )

        start = time.time()

        pool.starmap_async( calculate_sim_it1_it2,
                            [(it1, it2, e1, conf.host, conf.user, conf.password, conf.database) for it2 in train] ).get()
        end = time.time()
        print( end - start )
        pool.close()
        count += 1
        print( count )

# ----------------------------------------------------------------------------------------------------- #

def calculate_semantic_similarity_gpu(onto_ids, name_prefix):
    """
    Calculate semantic similarity 
    :param onto_ids: list of ids (int)
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return:
    """

    conf = cfg.getInstance()
   
    ssmpy.semantic_base( conf.path_to_ontology )

    count = 0
    for it1 in onto_ids:
        if name_prefix == 'HP_':
            it1 = f'{s:07d}' + str( it1 )
        it1_ = name_prefix + str( it1 )
        print( it1_ )
        e1 = ssmpy.get_id( it1_ )

        # pool = mp.Pool(mp.cpu_count())

        calculate_sim_it1_it2_test_gpu( it1, e1, onto_ids, name_prefix )

        mask = np.where( onto_ids == it1 )
        onto_ids = np.delete( onto_ids, mask )
        count += 1
        print( count )
