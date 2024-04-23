import implicit
import math
from recommender_evaluation import *
from cross_val import *
from semsimcalculus import *
import gc
from database import *

# ----------------------------------------------------------------------------------------------------- #

def recommendations(model, train_data, test_items, user):
    print("user: ", user)
    print("model user factors: ", model.user_factors.shape)
    user_items = train_data.T.tocsr()  # user, item, rating
    return model.rank_items( user, user_items, test_items )

# ----------------------------------------------------------------------------------------------------- #

def map_original_id_to_system_id(item_score, original_item_id):
    """
    map the original id to the system ids
    :param item_score:
    :param original_item_id:
    :return:
    """
    name_prefix = cfg.getInstance().item_prefix[:-1]
    iscore_ontology = item_score.rename( columns={"item": "item_" + name_prefix} )
    iscore_ontology["item"] = iscore_ontology["item_" + name_prefix].map(
        original_item_id.set_index( 'item' )["new_index"] ).fillna( 0 )
    return iscore_ontology

# ----------------------------------------------------------------------------------------------------- #

def map_system_id_to_original_id(item_score, original_item_id):
    """
    map the id to the original ids
    :param item_score:
    :param original_item_id:
    :return:
    """
    name_prefix = cfg.getInstance().item_prefix[:-1]
    item_score["item_" + name_prefix] = item_score["item"].map(
        original_item_id.set_index( 'new_index' )["item"] ).fillna( 0 )
    return item_score

# ----------------------------------------------------------------------------------------------------- #

def select_metric(scores_by_item, metric):
    """
    select the column with the metric to use
    :param scores_by_item: pd DataFrame with all compounds and metrics
    :param metric: metric to select to calculate the mean of the similarities
    :return: pd DataFrame with columns item, score
    """
    item_score = scores_by_item[['comp_1', metric]]

    # item_score = item_score.groupby(['comp_1']).sum().reset_index().sort_values(metric, ascending=False).head(5).mean()
    item_score = item_score.groupby( 'comp_1' ).apply(
        lambda x: x.sort_values( (metric), ascending=False ).head( cfg.getInstance().n ).mean() )

    item_score = item_score.rename( columns={'comp_1': 'item', metric: 'score'} )
    item_score.item = item_score.item.astype( int )

    return item_score

# ----------------------------------------------------------------------------------------------------- #

def onto_algorithm(train_ratings_for_t_us, test_items_onto_id, metric):
    """

    :param train_ratings_for_t_us:
    :param test_items_onto_id:
    :param metric:
    :return: pandas dataframe: columns = item, score (item with onto_id)
    """

    # get just the IDs of the items in the train set
    train_items_for_t_us = train_ratings_for_t_us.item.unique()
    #training items for this user to be used for finding the similarity
    # get the score for each item in the test set
    scores_by_item = get_score_by_item( test_items_onto_id, train_items_for_t_us )
    if len( scores_by_item ) == 0:
        return [], [], []
    iscore_lin, iscore_resnik, iscore_jc = [], [], []

    if metric in ('sim_lin', 'all'):
        iscore_lin = select_metric( scores_by_item, 'sim_lin' )
    if metric in ('sim_resnik', 'all'):
        iscore_resnik = select_metric( scores_by_item, 'sim_resnik' )
    if metric in ('sim_jc', 'all'):
        iscore_jc = select_metric( scores_by_item, 'sim_jc' )

    return iscore_lin, iscore_resnik, iscore_jc

# ----------------------------------------------------------------------------------------------------- #

def all_evaluation_metrics(item_score, ratings_t_us, test_items, relevant, metrics_dict):
    """
    calculate the top k (size of the list of recommendations) for all metrics:
    P, R, F, fpr, rr, nDCG, auc
    :param item_score:
    :param ratings_t_us:
    :param test_items:
    :param relevant:
    :param metrics_dict: list of
    :return: list of the top k with all metrics: P, R, F, fpr, rr, nDCG, auc
    """

    user_r = [0.0]
    user_fpr = [0.0]
    k = cfg.getInstance().topk
    for i in range( 1, k + 1 ):
        top_n = get_top_n( item_score, i )
        top_n.item = top_n.item.astype( int )

        topn_real_ratings = get_real_item_rating( top_n, ratings_t_us ).rating

        fpr = false_positive_rate( test_items, relevant, top_n )

        recs = np.array( top_n.item ).astype( int )
        P = precision( recs, np.array( relevant ) )
        R = recall( recs, np.array( relevant ) )
        F = fmeasure( P, R )
        rr = reciprocal_rank( topn_real_ratings )

        user_r.append( R )
        user_fpr.append( fpr )

        nDCG = ndcg_at_k( topn_real_ratings, i, method=0 )

        # auc = metrics.auc(user_fpr, user_r)
        auc = get_auc( user_r, user_fpr )

        if len( metrics_dict ) != k:
            metrics_dict.update( {'top' + str( i ): [P, R, F, fpr, rr, nDCG, auc]} )

        else:
            old = np.array( metrics_dict['top' + str( i )] )
            new = np.array( [P, R, F, fpr, rr, nDCG, auc] )

            to_update = old + new
            metrics_dict.update( {'top' + str( i ): to_update} )

    return metrics_dict

# ----------------------------------------------------------------------------------------------------- #

def get_evaluation(test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                   ratings_train_sparse_CF, test_items, all_ratings, original_item_id, metric):
    """
    evaluate the results for all the algorithms and all the metrics, before save it in
    the csv file
    :param test_users:
    :param test_users_size:
    :param count_cv:
    :param count_cv_items:
    :param ratings_test:
    :param ratings_train_sparse_CF:
    :param test_items:
    :param all_ratings:
    :param original_item_id:
    :return: metrics_dict* for all algorithms (and combinations) and all the metrics
    """

    # CB
    onto_lin = {}
    onto_resnik = {}
    onto_jc = {}

    # CF
    als = {}
    bpr = {}

    # Hybrid
    als_onto_lin_m1 = {}
    als_onto_resnik_m1 = {}
    als_onto_jc_m1 = {}
    bpr_onto_lin_m1 = {}
    bpr_onto_resnik_m1 = {}
    bpr_onto_jc_m1 = {}

    als_onto_lin_m2 = {}
    als_onto_resnik_m2 = {}
    als_onto_jc_m2 = {}
    bpr_onto_lin_m2 = {}
    bpr_onto_resnik_m2 = {}
    bpr_onto_jc_m2 = {}

    model_bayes = implicit.bpr.BayesianPersonalizedRanking( factors=150, num_threads=10, use_gpu=False )
    model_als = implicit.als.AlternatingLeastSquares( factors=150, num_threads=10, use_gpu=False )

    # print(ratings_train_sparse_CF)
    model_als.fit( ratings_train_sparse_CF )
    model_bayes.fit( ratings_train_sparse_CF )

    progress = 0
    users_to_remove = 0
    relevant_items_sum = 0

    # to use in onto algorithm
    test_items_onto_id = all_ratings[all_ratings.index_item.isin(
        test_items )].item.unique()

    for t_us in test_users:
        print("-- t_us --", t_us)
        progress += 1
        print( progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r" )

        sys.stdout.flush()

        # all ratings for user t_us (index_user)
        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]
        # train ratings for user t_us
        train_ratings_for_t_us_CB = all_ratings_for_t_us[
            ~(all_ratings_for_t_us.index_item.isin( ratings_test.index_item ))]

        # verify it user has condition to be evaluated, i.e., it has al least one item in the test set
        ratings_test_t_us = all_ratings_for_t_us[(all_ratings_for_t_us.index_item.isin( ratings_test.index_item ))]

        if np.sum( ratings_test_t_us.rating ) == 0:
            users_to_remove += 1
            continue

        if len( train_ratings_for_t_us_CB ) == 0:
            users_to_remove += 1
            continue

        iscore_lin, iscore_resnik, iscore_jc = onto_algorithm( train_ratings_for_t_us_CB,
                                                               test_items_onto_id, metric )

        # TO REMOVE PRINT
        if len( iscore_lin ) == 0:
            users_to_remove += 1

            with open( '../iscore_lin_empty.txt', 'a' ) as g:
                g.write( "item score: {item} \ntrain: {train}\n".
                    format(
                    item=iscore_lin,
                    train=train_ratings_for_t_us_CB )
                )
                g.close()
            """ print(iscore_lin)
            print('EMPTYYYY')
            print(train_ratings_for_t_us_CB) """
            continue

        if metric in ('sim_lin', 'all'):
            iscore_lin = map_original_id_to_system_id( iscore_lin, original_item_id )
        if metric in ('sim_resnik', 'all'):
            iscore_resnik = map_original_id_to_system_id( iscore_resnik, original_item_id )
        if metric in ('sim_jc', 'all'):
            iscore_jc = map_original_id_to_system_id( iscore_jc, original_item_id )

        iscore_implicit_als = get_score_by_implicit( model_als, ratings_train_sparse_CF, test_items, t_us )
        iscore_implicit_als = map_system_id_to_original_id( iscore_implicit_als, original_item_id )
        iscore_implicit_bpr = get_score_by_implicit( model_bayes, ratings_train_sparse_CF, test_items, t_us )
        iscore_implicit_bpr = map_system_id_to_original_id( iscore_implicit_bpr, original_item_id )

        # print('onto lin')
        # print(iscore_lin.sort_values(by=['score'], ascending=False).head(20))
        # print('onto resnik')
        # print(iscore_resnik.sort_values(by=['score'], ascending=False).head(20))
        # print('onto jc')
        # print(iscore_jc.sort_values(by=['score'], ascending=False).head(20))
        # print('als')
        # print(iscore_implicit_als.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr')
        # print(iscore_implicit_bpr.sort_values(by=['score'], ascending=False).head(20))

        # Different merge scores will be used:
        #   1: product
        #   2: arithmetic
        #   3: quadratic
        #   4: harmonic
      
        iscore_als_onto_lin_m1 = merge_algorithms_scores( iscore_lin, iscore_implicit_als, 1 )
        iscore_als_onto_lin_m2 = merge_algorithms_scores( iscore_lin, iscore_implicit_als, 2 )
        iscore_bpr_onto_lin_m1 = merge_algorithms_scores( iscore_lin, iscore_implicit_bpr, 1 )
        iscore_bpr_onto_lin_m2 = merge_algorithms_scores( iscore_lin, iscore_implicit_bpr, 2 )

        # print('als_onto_lin_m1')
        # print(iscore_als_onto_lin_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('als_onto_lin_m2')
        # print(iscore_als_onto_lin_m2.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_lin_m1')
        # print(iscore_bpr_onto_lin_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_lin_m2')
        # print(iscore_bpr_onto_lin_m2.sort_values(by=['score'], ascending=False).head(20))

        if metric in ('sim_resnik', 'all'):
            iscore_als_onto_resnik_m1 = merge_algorithms_scores( iscore_resnik, iscore_implicit_als, 1 )
            iscore_als_onto_resnik_m2 = merge_algorithms_scores( iscore_resnik, iscore_implicit_als, 2 )
            iscore_bpr_onto_resnik_m1 = merge_algorithms_scores( iscore_resnik, iscore_implicit_bpr, 1 )
            iscore_bpr_onto_resnik_m2 = merge_algorithms_scores( iscore_resnik, iscore_implicit_bpr, 2 )

        # print('als_onto_resnik_m1')
        # print(iscore_als_onto_resnik_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('als_onto_resnik_m2')
        # print(iscore_als_onto_resnik_m2.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_resnik_m1')
        # print(iscore_bpr_onto_resnik_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_resnik_m2')
        # print(iscore_bpr_onto_resnik_m2.sort_values(by=['score'], ascending=False).head(20))

        if metric in ('sim_jc', 'all'):
            iscore_als_onto_jc_m1 = merge_algorithms_scores( iscore_jc, iscore_implicit_als, 1 )
            iscore_als_onto_jc_m2 = merge_algorithms_scores( iscore_jc, iscore_implicit_als, 2 )
            iscore_bpr_onto_jc_m1 = merge_algorithms_scores( iscore_jc, iscore_implicit_bpr, 1 )
            iscore_bpr_onto_jc_m2 = merge_algorithms_scores( iscore_jc, iscore_implicit_bpr, 2 )

        # print('als_onto_jc_m1')
        # print(iscore_als_onto_jc_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('als_onto_jc_m2')
        # print(iscore_als_onto_jc_m2.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_jc_m1')
        # print(iscore_bpr_onto_jc_m1.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_onto_jc_m2')
        # print(iscore_bpr_onto_jc_m2.sort_values(by=['score'], ascending=False).head(20))

        relevant = get_relevants_by_user( ratings_test_t_us, 0 )
        # print("relevant: ", relevant)

        relevant_items_sum += len( relevant )  # so esta a fazer media

        als = all_evaluation_metrics( iscore_implicit_als, ratings_test_t_us, test_items,
                                      relevant.index_item, als )
        bpr = all_evaluation_metrics( iscore_implicit_bpr, ratings_test_t_us, test_items,
                                      relevant.index_item, bpr )

        if metric in ('sim_lin', 'all'):
            # hybrid 
            onto_lin = all_evaluation_metrics( iscore_lin, ratings_test_t_us, test_items,
                                               relevant.index_item, onto_lin )
            # hybrid metric 1
            als_onto_lin_m1 = all_evaluation_metrics( iscore_als_onto_lin_m1,
                                                      ratings_test_t_us, test_items, relevant.index_item,
                                                      als_onto_lin_m1 )
            bpr_onto_lin_m1 = all_evaluation_metrics( iscore_bpr_onto_lin_m1,
                                                      ratings_test_t_us, test_items, relevant.index_item,
                                                      bpr_onto_lin_m1 )
            # hybrid metric 2
            als_onto_lin_m2 = all_evaluation_metrics( iscore_als_onto_lin_m2,
                                                      ratings_test_t_us, test_items, relevant.index_item,
                                                      als_onto_lin_m2 )
            bpr_onto_lin_m2 = all_evaluation_metrics( iscore_bpr_onto_lin_m2,
                                                      ratings_test_t_us, test_items, relevant.index_item,
                                                      bpr_onto_lin_m2 )

        if metric in ('sim_resnik', 'all'):
            # hybrid 
            onto_resnik = all_evaluation_metrics( iscore_resnik, ratings_test_t_us, test_items,
                                                  relevant.index_item,
                                                  onto_resnik )
            # hybrid metric 1
            als_onto_resnik_m1 = all_evaluation_metrics( iscore_als_onto_resnik_m1,
                                                         ratings_test_t_us, test_items,
                                                         relevant.index_item,
                                                         als_onto_resnik_m1 )
            bpr_onto_resnik_m1 = all_evaluation_metrics( iscore_bpr_onto_resnik_m1,
                                                         ratings_test_t_us, test_items,
                                                         relevant.index_item,
                                                         bpr_onto_resnik_m1 )

            # hybrid metric 2       
            als_onto_resnik_m2 = all_evaluation_metrics( iscore_als_onto_resnik_m2,
                                                         ratings_test_t_us, test_items,
                                                         relevant.index_item,
                                                         als_onto_resnik_m2 )
            bpr_onto_resnik_m2 = all_evaluation_metrics( iscore_bpr_onto_resnik_m2,
                                                         ratings_test_t_us, test_items,
                                                         relevant.index_item,
                                                         bpr_onto_resnik_m2 )
        if metric in ('sim_jc', 'all'):
            # hybrid
            onto_jc = all_evaluation_metrics( iscore_jc, ratings_test_t_us, test_items,
                                              relevant.index_item,
                                              onto_jc )
            # hybrid metric 1
            als_onto_jc_m1 = all_evaluation_metrics( iscore_als_onto_jc_m1,
                                                     ratings_test_t_us, test_items,
                                                     relevant.index_item,
                                                     als_onto_jc_m1 )
            bpr_onto_jc_m1 = all_evaluation_metrics( iscore_bpr_onto_jc_m1,
                                                     ratings_test_t_us, test_items,
                                                     relevant.index_item,
                                                     bpr_onto_jc_m1 )
            # hybrid metric 2  
            als_onto_jc_m2 = all_evaluation_metrics( iscore_als_onto_jc_m2,
                                                     ratings_test_t_us, test_items,
                                                     relevant.index_item,
                                                     als_onto_jc_m2 )
            bpr_onto_jc_m2 = all_evaluation_metrics( iscore_bpr_onto_jc_m2,
                                                     ratings_test_t_us, test_items,
                                                     relevant.index_item,
                                                     bpr_onto_jc_m2 )

    print( "users size: ", test_users_size )
    test_users_size = test_users_size - users_to_remove

    print( "n users removed: ", users_to_remove )

    relevant_items_mean = relevant_items_sum / test_users_size

    print( "mean of relevant items: ", relevant_items_mean )

    als = calculate_dictionary_mean( als, float( test_users_size ) )
    bpr = calculate_dictionary_mean( bpr, float( test_users_size ) )

    if metric in ('sim_lin', 'all'):
        onto_lin = calculate_dictionary_mean( onto_lin, float( test_users_size ) )
        als_onto_lin_m1 = calculate_dictionary_mean( als_onto_lin_m1, float( test_users_size ) )
        bpr_onto_lin_m1 = calculate_dictionary_mean( bpr_onto_lin_m1, float( test_users_size ) )
        als_onto_lin_m2 = calculate_dictionary_mean( als_onto_lin_m2, float( test_users_size ) )
        bpr_onto_lin_m2 = calculate_dictionary_mean( bpr_onto_lin_m2, float( test_users_size ) )

    if metric in ('sim_resnik', 'all'):
        onto_resnik = calculate_dictionary_mean( onto_resnik, float( test_users_size ) )
        als_onto_resnik_m1 = calculate_dictionary_mean( als_onto_resnik_m1, float( test_users_size ) )
        bpr_onto_resnik_m1 = calculate_dictionary_mean( bpr_onto_resnik_m1, float( test_users_size ) )
        als_onto_resnik_m2 = calculate_dictionary_mean( als_onto_resnik_m2, float( test_users_size ) )
        bpr_onto_resnik_m2 = calculate_dictionary_mean( bpr_onto_resnik_m2, float( test_users_size ) )

    if metric in ('sim_jc', 'all'):
        onto_jc = calculate_dictionary_mean( onto_jc, float( test_users_size ) )
        als_onto_jc_m1 = calculate_dictionary_mean( als_onto_jc_m1, float( test_users_size ) )
        bpr_onto_jc_m1 = calculate_dictionary_mean( bpr_onto_jc_m1, float( test_users_size ) )
        als_onto_jc_m2 = calculate_dictionary_mean( als_onto_jc_m2, float( test_users_size ) )
        bpr_onto_jc_m2 = calculate_dictionary_mean( bpr_onto_jc_m2, float( test_users_size ) )

    del model_bayes
    del model_als
    gc.collect()

    return onto_lin, onto_resnik, onto_jc, als, bpr, \
           als_onto_lin_m1, als_onto_resnik_m1, als_onto_jc_m1, \
           bpr_onto_lin_m1, bpr_onto_resnik_m1, bpr_onto_jc_m1, \
           als_onto_lin_m2, als_onto_resnik_m2, als_onto_jc_m2, \
           bpr_onto_lin_m2, bpr_onto_resnik_m2, bpr_onto_jc_m2

# ----------------------------------------------------------------------------------------------------- #

def get_all_metrics_by_cv_implicit(test_users, test_users_size, count_cv, count_cv_items, ratings_test, model_als,
                                   ratings_sparse, test_items, train_ratings):
    metrics_dict = {}

    progress = 0
    users_to_remove = 0
    relevant_items_sum = 0

    for t_us in test_users:

        progress += 1
        print( progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r" )

        sys.stdout.flush()

        ratings_t_us = ratings_test[ratings_test.user == t_us]
        train_ratings_t_us = train_ratings[train_ratings.user == t_us]

        if np.sum( ratings_t_us.rating ) == 0:
            users_to_remove += 1
            continue

        if len( train_ratings_t_us ) == 0:
            users_to_remove += 1
            continue

        relevant = get_relevants_by_user( ratings_t_us, 0 )
        relevant_items_sum += len( relevant )
        print("relevant: ", relevant)
        print("---------------------Debug part---------------------")
        item_score = recommendations( model_als, ratings_sparse, test_items, t_us )
        item_score = pd.DataFrame( np.array( item_score ), columns=["item", "score"] )

        metrics_dict = all_evaluation_metrics( item_score, ratings_t_us, test_items, relevant, metrics_dict )

    return metrics_dict_aux( test_users_size, users_to_remove, relevant_items_sum, metrics_dict )

# ----------------------------------------------------------------------------------------------------- #

def metrics_dict_aux(users_size, users_to_remove, relevant, metrics):
    #users_size = users_size - users_to_remove
    #relevant_mean = relevant / users_size
    # print("mean of relevant items: ", relevant_mean, " n users removed: ", users_to_remove)
    return calculate_dictionary_mean( metrics, float( users_size - users_to_remove ) )


# ----------------------------------------------------------------------------------------------------- #

def get_score_by_item(test_items_onto_id, train_items_for_t_us):
    sims = get_read_all( test_items_onto_id, train_items_for_t_us )
    sims_inverse = get_read_all( train_items_for_t_us, test_items_onto_id )
    sims_inverse = sims_inverse.rename( columns={"comp_1": "comp_2", "comp_2": "comp_1"} )

    sims_concat = pd.concat( [sims, sims_inverse], axis=0, join='outer', ignore_index=True, sort=False )

    if len( sims_concat ) > 0:
        # scores_by_item = sims_concat.groupby(['comp_1']).mean().reset_index()
        return sims_concat
    else:
        return []

# ----------------------------------------------------------------------------------------------------- #

def get_all_metrics_by_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                                test_items, all_ratings):
    metrics_dict = {}
    progress = 0
    users_to_remove = 0
    relevant_items_sum = 0

    #test items onto id!!!  what i'm rating. Array is equal for all users
    test_items_onto_id = all_ratings[all_ratings.index_item.isin(
        test_items )].item.unique()  

    for t_us in test_users:

        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]

        test_ratings_for_t_us = all_ratings_for_t_us[all_ratings_for_t_us.index_item.isin( test_items )]
        train_ratings_for_t_us = all_ratings_for_t_us[~(all_ratings_for_t_us.index_item.isin( test_items ))]

        # training items for this user to be used for finding the similarity
        train_items_for_t_us = train_ratings_for_t_us.item.unique()  

        progress += 1
        print( progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r" )

        sys.stdout.flush()

        ratings_t_us = ratings_test[ratings_test.user == t_us]

        if np.sum( ratings_t_us.rating ) == 0:
            users_to_remove += 1

            continue

        if len( train_ratings_for_t_us ) == 0:
            users_to_remove += 1

            continue

        scores_by_item = get_score_by_item( test_items_onto_id, train_items_for_t_us )

        relevant = get_relevants_by_user( test_ratings_for_t_us, 0 )

        relevant_items_sum += len( relevant )

        if len( scores_by_item ) > 0:
            # item_score = recommendations(model_als, ratings_train_sparse_CF, test_items, t_us)
            # item_score = pd.DataFrame(np.array(item_score), columns=["item", "score"])

            sim_metric = cfg.getInstance().sim_metric
            item_score = scores_by_item[['comp_1', sim_metric]]
            item_score = item_score.rename( columns={"comp_1": "item", sim_metric: "score"} )
            item_score.item = item_score.item.astype( int )

            metrics_dict = all_evaluation_metrics( item_score, test_ratings_for_t_us, test_items_onto_id,
                                                   relevant, metrics_dict )

    return metrics_dict_aux( test_users_size, users_to_remove, relevant_items_sum, metrics_dict )

# ----------------------------------------------------------------------------------------------------- #

def merge_algorithms_scores(iscore_ontology, iscore_implicit, metric):
    """
    calculates the scores for each test item with hybrid algorithm
    :param iscore_ontology: item score from CB
    :param iscore_implicit: item score from CF
    :param metric: 1: multiplication of the scores; 2: mean of the scores
    :return: item score dataframe order descending
    """

    merged_item_scores = pd.merge( iscore_implicit, iscore_ontology, on='item' )

    if metric == 1:
        merged_item_scores['score'] = merged_item_scores.score_x * merged_item_scores.score_y

    elif metric == 2: #'arithmetic'
        merged_item_scores['score'] = (merged_item_scores.score_x + merged_item_scores.score_y) / 2

    elif metric == 3: #'quadratic'
        merged_item_scores['score'] = math.sqrt(merged_item_scores.score_x**2 + merged_item_scores.score_y**2) / 2 

    elif metric == 4: #'harmonic'
        merged_item_scores['score'] = 2 / ( 1/merged_item_scores.score_x + 1/merged_item_scores.score_y)


    return merged_item_scores[['item', 'score', 'item_' + cfg.getInstance().item_prefix + 'x']].sort_values(
        by=['score'], ascending=False )

# ----------------------------------------------------------------------------------------------------- #

def hybrid_implicit_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test, model_als,
                             ratings_sparse, test_items, train_ratings, all_ratings, original_item_id):
    """
    
    :param
    :param
    :param sim_metric: specify the similarity metric (sim_lin, sim_resnik or sim_jc)
    :return: item score dataframe order descending
    """
    print("Not sure the error is here!")
    metrics_dict = {}

    progress = 0

    users_to_remove = 0
    relevant_items_sum = 0
    # ssmpy.semantic_base("/mlData/X.db")

    # test items onto id!!!  what i'm rating. Array is equal for all users
    test_items_onto_id = all_ratings[all_ratings.index_item.isin(
        test_items )].item.unique()  

    for t_us in test_users:

        progress += 1
        print( progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r" )

        sys.stdout.flush()

        # implicit
        ratings_t_us = ratings_test[ratings_test.user == t_us]
        train_ratings_t_us = train_ratings[train_ratings.user == t_us]

        # ontology
        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]
        test_ratings_for_t_us = all_ratings_for_t_us[all_ratings_for_t_us.index_item.isin( test_items )]
        train_ratings_for_t_us = all_ratings_for_t_us[~(all_ratings_for_t_us.index_item.isin( test_items ))]
        # training items for this user to be user for finding the similarity
        train_items_for_t_us = train_ratings_for_t_us.item.unique()  

        if np.sum( ratings_t_us.rating ) == 0:
            users_to_remove += 1

            continue

        if len( train_ratings_t_us ) == 0:
            users_to_remove += 1

            continue
        print("--------------DEBUG ------------------------------#################")
        iscore_implicit = get_score_by_implicit( model_als, ratings_sparse, test_items, t_us )
        iscore_ontology = get_score_by_ontology( test_items_onto_id, train_items_for_t_us )

        item_score = merge_algorithms_scores( iscore_ontology, original_item_id, iscore_implicit )

        relevant = get_relevants_by_user( ratings_t_us, 0 )
        relevant_items_sum += len( relevant )

        metrics_dict = all_evaluation_metrics( item_score, ratings_t_us, test_items, relevant, metrics_dict )

    return metrics_dict_aux( test_users_size, users_to_remove, relevant_items_sum, metrics_dict )

# ----------------------------------------------------------------------------------------------------- #

# def get_score_by_implicit(model, ratings_sparse, test_items, t_us):
#     item_score = recommendations( model, ratings_sparse, test_items, t_us )
#     print("item score", item_score)
#     # Assuming item_score is a tuple of two arrays: (array of item IDs, array of scores)
#     item_ids, scores = item_score  # Unpack the tuple into two separate arrays

#     # Combine these arrays into a list of tuples
#     combined = list(zip(item_ids, scores))

#     # Create a DataFrame from the combined list
#     item_score = pd.DataFrame(combined, columns=["item", "score"])

#     print("item score 2",item_score)
#     # item_score = pd.DataFrame( np.array( item_score ), columns=["item", "score"] )
#     item_score.item = item_score.item.astype( int )

#     return item_score
def get_score_by_implicit(model, ratings_sparse, test_items, t_us):
    # Before making a call to recommendations, check if t_us is within the valid range
    # if t_us >= model.user_factors.shape[0]:
    #     print(f"User ID {t_us} is out of bounds.")
    #     # Return an empty DataFrame or some default value in case of out of bounds
    #     return pd.DataFrame([], columns=["item", "score"])
    
    item_score = recommendations(model, ratings_sparse, test_items, t_us)
    # Assuming item_score is a tuple of two arrays: (array of item IDs, array of scores)
    item_ids, scores = item_score  # Unpack the tuple into two separate arrays

    # Combine these arrays into a list of tuples
    combined = list(zip(item_ids, scores))

    # Create a DataFrame from the combined list
    item_score = pd.DataFrame(combined, columns=["item", "score"])
    item_score.item = item_score.item.astype(int)

    return item_score


# ----------------------------------------------------------------------------------------------------- #

def get_score_by_ontology(test_items_onto_id, train_items_for_t_us):
    scores_by_item = get_score_by_item( test_items_onto_id, train_items_for_t_us )

    # item_score = scores_by_item[['comp_1', 'sim_lin']]
    # item_score = item_score.rename(columns={"comp_1": "item", "sim_lin": "score"})

    sim_metric = cfg.getInstance().sim_metric
    item_score = scores_by_item[['comp_1', sim_metric]]
    item_score = item_score.rename( columns={'comp_1': 'item', sim_metric: 'score'} )
    item_score.item = item_score.item.astype( int )

    return item_score