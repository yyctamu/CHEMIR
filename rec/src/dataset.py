import pandas as pd
import numpy as np

def upload_dataset(csv_path, name_prefix):
    '''

    :param csv_path: <user, item, rating, ... > csv file
    :param name_prefix: prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return: user, item, rating pandas dataframe
    '''
       
    matrix = pd.read_csv( csv_path, sep=',' )
    
    if ( len(matrix.columns) > 3 ):
        matrix.columns = ['user_name', 'item', 'rating', 'user', 'item_label']
        matrix = matrix[['user', 'item', 'rating']]
    else:
        matrix.columns = ['user', 'item', 'rating']

    if ( matrix.dtypes['item'] == object ):
        # filter rows for specific ontology
        matrix = matrix[matrix['item'].astype(str).str.startswith( name_prefix )]
        # filter by number of users
        #matrix = matrix.groupby( 'user' ).filter( lambda x: len( x ) > 19 )
        # remove acronym of ontology and convert as int
        matrix['item'] = matrix['item'].str.replace( name_prefix, '' ).astype( int )
    
    return matrix