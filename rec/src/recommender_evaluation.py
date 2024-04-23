import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics


# ----------------------------------------------------------------------------------------------------- #

def get_relevants_by_user(df, threshold):
    """
    Get relevants user accordling some threshold
    :param df: pandas DataFrame
    :param threshold: minimum value in the ontology
    """
    return df[df.rating >= threshold]


# ----------------------------------------------------------------------------------------------------- #

def get_top_n(items_scores, n):
    """
    Get top@k
    :param items_scores: 
    :param n: n for top@k
    :return First n items scores
    """
    items_scores = items_scores.sort_values( by=['score'], ascending=False )
    return items_scores.head( n )


# ----------------------------------------------------------------------------------------------------- #
# Precision

def precision(recomendations, relevant):
    mask = np.isin( recomendations, relevant )
    return ( len( recomendations[mask] ) / len( recomendations ) )


# ----------------------------------------------------------------------------------------------------- #
# Recall

def recall(recomendations, relevant):
    mask = np.isin( recomendations, relevant )
    if len( relevant != 0 ):
        return ( len( recomendations[mask] ) / len( relevant ) )
    else:
        return 0


# ----------------------------------------------------------------------------------------------------- #
# F-Measure

def fmeasure(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    else:
        return ( 2 * ((precision * recall) / (precision + recall)) )


# ----------------------------------------------------------------------------------------------------- #
# dcg

def get_real_item_rating(rank, user_ratings):
    # map the items to the rating given by the user
    rank["rating"] = rank["item"].map( user_ratings.set_index( 'index_item' )["rating"] ).fillna( 0 )
    return rank


def dcg_at_k(r, k, method=0):
    r = np.asfarray( r )[:k]
    if r.size:
        if method == 0:
            return  ( r[0] + np.sum( r[1:] / np.log2( np.arange( 2, r.size + 1 ) ) ) ) 
        elif method == 1:
            return np.sum( r / np.log2( np.arange( 2, r.size + 2 ) ) )
        else:
            raise ValueError( 'method must be 0 or 1.' )
    return 0.


# ----------------------------------------------------------------------------------------------------- #
# NDCG

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k( sorted( r, reverse=True ), k, method )
    if not dcg_max:
        return 0.
    return  ( dcg_at_k( r, k, method ) / dcg_max )


# ----------------------------------------------------------------------------------------------------- #
# Reciprocal rank

def reciprocal_rank(rs):
    if len( rs[rs != 0] > 0 ):
        first_nonzero_position = rs.to_numpy().nonzero()[0][0] + 1
        # print(first_nonzero_position)
        return ( 1 / first_nonzero_position )
    else:
        return 0


# ----------------------------------------------------------------------------------------------------- #
# False positive rate

def false_positive_rate(list_of_test_items, relevants, rank):
    """

    :param list_of_test_items: all candidates items to be recommended
    :param relevants: all relevants items in the test set
    :param rank: listed ranked by the algorithm
    :return: false positive rate
    """

    mask = np.isin( list_of_test_items, relevants )
    all_negatives = list_of_test_items[~mask]
    mask2 = np.isin( rank.item, relevants )
    fp = rank.item[~mask2]
    if len( all_negatives ) != 0:
        return ( len( fp ) / len( all_negatives ) )
    else:
        return 0


# ----------------------------------------------------------------------------------------------------- #
# AUC

def get_auc(recall, fpr):
    # add point 1,1
    recall.append( 1 )
    fpr.append( 1 )

    if np.sum( np.array( fpr ) ) == 0:
        auc = 1
    else:
        auc = metrics.auc( fpr, recall )
    del recall[-1]
    del fpr[-1]
    return auc


def auc_(recall, fpr, max_n):
    auc_list = []

    for a in range( 1, max_n ):
        recall_ = recall[0:a + 1]
        fpr_ = fpr[0:a + 1]
        auc = metrics.auc( fpr_, recall_ )
        auc_list.append( auc )
    # print(auc_list)
    return np.array( auc_list )


# ----------------------------------------------------------------------------------------------------- #
# RMSE

def rmse(predictions_list, real_list):
    return predict.rmse( predictions_list, real_list, missing='ignore' )


# ----------------------------------------------------------------------------------------------------- #
# MAE

def mae(predictions_list, real_list):
    return predict.mae( predictions_list, real_list, missing='ignore' )


# ----------------------------------------------------------------------------------------------------- #
# Normalize

def normalize_between_range(array, a, b):
    return ( ((b - a) * ((array - array.min()) / (array.max() - array.min()))) + a )


# ----------------------------------------------------------------------------------------------------- #
# Top k metrics sum

def topk_metrics_sum(P, R, F, rr, nDCG, n):
    my_file = Path( "temp" + str( n ) + ".csv" )
    if my_file.is_file():

        df = pd.read_csv( my_file, sep=',', header=None )
        df_array = np.array( df )
        df_array[0] = np.add( np.array( [P, R, F, rr, nDCG] ), df_array[0] )
        #        print(df_array)
        np.savetxt( "temp" + str( n ) + ".csv", df_array, delimiter="," )
    else:
        line = np.array( [[P, R, F, rr, nDCG]] )
        np.savetxt( "temp" + str( n ) + ".csv", line, delimiter="," )
        # line.to_csv("mlData/temp" + str(n) + ".csv")
