import pandas as pd
import numpy as np
from itertools import product
from sqlalchemy import create_engine, select
import mysql.connector as connector
import sqlite3
from sqlite3 import Error
from myconfiguration import MyConfiguration as cfg


# ----------------------------------------------------------------------------------------------------- #

def create_default_connection_mysql():
    """
    create a default connection to the mysql database
        specified by host, user and password defined in config.ini
    :param
    :return: Connection object
    """
    conf = cfg.getInstance()
    assert isinstance( conf.password,object )
    mydb = connector.connect(
        host=conf.host,
        user=conf.user,
        password=conf.password,
        use_pure=True,
        ssl_disabled=True
    )
    return mydb


# ----------------------------------------------------------------------------------------------------- #

def create_connection_mysql():
    """ create a connection to the mysql database
       specified by host, user, password and
       database name defined in config.ini
   :param
   :return: Connection object
   """
    mydb = create_default_connection_mysql()
    mydb.database = cfg.getInstance().database

    return mydb


# ----------------------------------------------------------------------------------------------------- #

def create_connection_sqlite(sb_file):
    """ 
    
     Create a database connection to the SQLite database
    specified by sb_file
    :param sb_file: sqlite database filename
    :type sb_file: string
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect( sb_file )
        return conn
    except Error as e:
        print( e )

    return None


# ----------------------------------------------------------------------------------------------------- #

def create_engine_mysql():
    """
    
    Create a pool and dialect together connection to provide a source of database
    and behavior
    :param:
    :return: Connection engine
    """
    # in case of connection error, change the host as in the next commented code
    conf = cfg.getInstance()
    host = conf.host
    user = conf.user
    passwd = conf.password
    db_name = conf.database

    return create_engine( "mysql+pymysql://{user}:{pw}@{host}/{db}"
                          .format( user=user,
                                   pw=passwd,
                                   host=host,
                                   db=db_name ),
                          pool_pre_ping=True )


# ----------------------------------------------------------------------------------------------------- #

def check_database():
    """
    check the existence of a database with the name defined in config.ini
    if none, a new is created as well as a table of similarity
    :param: none
    :return:
    """
    global mydb, mycursor
    try:
        check = False
        mydb = create_default_connection_mysql()

        mycursor = mydb.cursor()

        db_name = cfg.getInstance().database

        mycursor.execute( "SHOW DATABASES" )
        for x in mycursor:
            # x = x[0].decode("unicode-escape") # decode was giving an error (note: decode is needed when using mysql docker)

            if x[0].decode( "unicode-escape" ) == db_name:
                # x[0].encode().decode('utf-8') == db_name:
                check = True

        if check != False:
            print( "Database already exists" )

        else:
            print( "The database is not available. Calculate similarity based "+
            "on SemanticSimDBCreator project" )

    except Error as e:
        print( "Error while connecting to MySQL", e )
    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()


# ----------------------------------------------------------------------------------------------------- #

def insert_row(it1, it2, sim_res, sim_l, sim_j):
    """
    insert rows in a similarity table with the corresponding values
    :param it1: entity 1 (id)
    :param it2: entity 2 (id)
    :param sim_res: Resnik semantic similarity
    :param sim_l: Lin's semantic similarity
    :param sim_j: Jiang and Conrath's semantic similarity
    :return:
    """

    global mydb, mycursor
    try:
        mydb = create_connection_mysql()

        mycursor = mydb.cursor()

        sql = "INSERT INTO similarity (comp_1, comp_2, sim_resnik, sim_lin, sim_jc) VALUES (%s,%s,%s,%s,%s)"

        val = (it1, it2, sim_res, sim_l, sim_j)
        mycursor.execute( sql, val )

        mydb.commit()
    except Error as e:
        print( "Error while connecting to MySQL", e )
    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()


def get_sim_where_comp(it1, it2):
    """
    Get similarity between it1 and it2
    :param it1: entity 1 (id)
    :param it2: entity 2 (id)
    """
    global len_my_cursor, mydb, mycursor
    try:
        mydb = create_connection_mysql()

        mycursor = mydb.cursor()
        sql = "select * from similarity where comp_1 = %s and comp_2 = %s"
        sql = sql % (it1, it2)
        mycursor.execute( sql )

        my_cursor = mycursor.fetchall()
        len_my_cursor = len( my_cursor )

    except Error as e:
        print( "Error while connecting to MySQL", e )
    finally:
        if (mydb.is_connected()):
            mycursor.close()
            mydb.close()

    return len_my_cursor

# ----------------------------------------------------------------------------------------------------- #

def check_if_pair_exist(it1, it2):
    """
    Check if similarity between ontology list exist
    :param it1: entity 1 (id)
    :param it2: entity 2 (id)
    :return: true or false
    """
    print( it1, it2 )

    exist = get_sim_where_comp( it1, it2 )
    exist_reverse = get_sim_where_comp( it2, it1 )
    if exist == 0 and exist_reverse == 0:
        return False

    else:

        return True

# ----------------------------------------------------------------------------------------------------- #

def get_items_ids(dataset, name_prefix):
    """

    :param dataset: dataset
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    """
    dataset.item = dataset.item.map( lambda x: x.lstrip( name_prefix ) ).astype( int )
    ids = dataset.item.unique()

    return ids

# ----------------------------------------------------------------------------------------------------- #

def confirm_all_test_train_similarities_create(list1, list2, scores_by_item, name_prefix):
    """
    
    :param list1: test_items of onto_id
    :param list2: train_items_for_t_us
    :param scores_by_item:
    :param name_prefix: prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return:
    """
    # test_items_onto_id = np.insert(test_items_onto_id, 1, 10000)
    print( "items in train", len( list2 ) )

    # check if all item-item pair was found in the database
    lists_combinations = pd.DataFrame( list( product( list1, list2 ) ),
                                       columns=['l1', 'l2'] )

    ss = lists_combinations.l1.isin(
        scores_by_item.comp_1.astype( 'int64' ).tolist() ) & lists_combinations.l2.isin(
        scores_by_item.comp_2.astype( 'int64' ).tolist() )

    ss2 = lists_combinations.l2.isin(
        scores_by_item.comp_1.astype( 'int64' ).tolist() ) & lists_combinations.l1.isin(
        scores_by_item.comp_2.astype( 'int64' ).tolist() )

    not_found_in_db = lists_combinations[(~ss) & (~ss2)]

    unique_test = not_found_in_db.l1.unique()
    unique_train = not_found_in_db.l2.unique()

    if len( unique_test ) > 0:
        calculate_semantic_similarity( unique_test, name_prefix, unique_train )
    else:
        print( "all items in DB" )

# ----------------------------------------------------------------------------------------------------- #

def confirm_all_test_train_similarities(list1, list2, pairs_from_db):
    # check if all item-item pair was found in the database

    lists_combinations = pd.DataFrame( list( product( list1, list2 ) ),
                                       columns=['l1', 'l2'] )

    ss = lists_combinations.l1.isin(
        pairs_from_db.comp_1.astype( 'int64' ).tolist() ) & lists_combinations.l2.isin(
        pairs_from_db.comp_2.astype( 'int64' ).tolist() )

    ss2 = lists_combinations.l2.isin(
        pairs_from_db.comp_1.astype( 'int64' ).tolist() ) & lists_combinations.l1.isin(
        pairs_from_db.comp_2.astype( 'int64' ).tolist() )

    not_found_in_db = lists_combinations[(~ss) & (~ss2)]

    not_found_list_1 = not_found_in_db.l1.unique().tolist()
    not_found_list_2 = not_found_in_db.l2.unique().tolist()

    return not_found_list_1, not_found_list_2

# ----------------------------------------------------------------------------------------------------- #

def get_sims(entry_ids_1, entry_ids_2):
    """
    Return <id, comp_1, comp_2> from database between 2 list of entries
    :param entry_ids_1: list of entries 1
    :param entry_ids_2: list of entries 2
    :return result: pandas Dataframe
    """
    global mydb, result

    try:
        mydb = create_connection_mysql()
        mycursor = mydb.cursor()

        format_strings1 = ','.join( ['%s'] * len( entry_ids_1 ) )
        format_strings2 = ','.join( ['%s'] * len( entry_ids_2 ) )
        sql = "select id, comp_1, comp_2 from similarity where comp_1 in (%s) and comp_2 in (%s)"
        format_strings1 = format_strings1 % tuple( entry_ids_1 )
        format_strings2 = format_strings2 % tuple( entry_ids_2 )
        sql = sql % (format_strings1, format_strings2)

        mycursor.execute( sql )

        result = mycursor.fetchall()

        if len( result ) != 0:

            result = pd.DataFrame( np.array( result ),
                                   columns=['id', 'comp_1', 'comp_2'] )

        else:
            result = pd.DataFrame( columns=['id', 'comp_1', 'comp_2'] )

    except Error as e:
        print( "Error while connecting to MySQL", e )
    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()

    return result

# ----------------------------------------------------------------------------------------------------- #

def get_read_all(entry_ids_1, entry_ids_2):
    """
    Return all columns from database between 2 list of entries
    :param entry_ids_1: list of entries 1
    :param entry_ids_2: list of entries 2
    :return result: pandas Dataframe

    """

    global mydb, result
    list1 = entry_ids_1.tolist()
    list2 = entry_ids_2.tolist()
    try:
        mydb = create_connection_mysql()

        format_strings1 = ','.join( ['%s'] * len( list1 ) )
        format_strings2 = ','.join( ['%s'] * len( list2 ) )
        sql = "select * from similarity where comp_1 in (%s) and comp_2 in (%s)"
        format_strings1 = format_strings1 % tuple( list1 )
        format_strings2 = format_strings2 % tuple( list2 )
        sql = sql % (format_strings1, format_strings2)

        result = pd.read_sql_query( sql, con=mydb )

    except Error as e:
        print( "Error while connecting to MySQL", e )
    finally:
        if mydb.is_connected():
            mydb.close()

    return result


# ----------------------------------------------------------------------------------------------------- #

def save_to_mysql(df, engine, table_name, name_prefix):
    """

    :param df: pandas Dataframe
    :param engine: engine object
    :param table_name: name of table where results are saved
    :param name_prefix: Prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    """

    df.comp_1 = df.comp_1.map( lambda x: x.lstrip( name_prefix ) ).astype( int )
    df.comp_2 = df.comp_2.map( lambda x: x.lstrip( name_prefix ) ).astype( int )

    df.to_sql( table_name, con=engine, if_exists='append', index=False, method='multi', chunksize=10000 )
