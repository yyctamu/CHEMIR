import configargparse


class MyConfiguration:
    __instance = None

    @staticmethod
    def getInstance() -> object:
        """ Static access method. """
        if MyConfiguration.__instance is None:
            p = configargparse.ArgParser( default_config_files=['../config/config.ini'] )

            p.add( '-mc', '--my-config', is_config_file=True, help='alternative config file path' )

            p.add( "-ds", "--path_to_dataset", required=False, help="path to dataset", type=str )

            p.add( "-cv", "--cv", required=False, help="cross validation folds",
                   type=int )

            p.add( "-k", "--topk", required=False, help="k for topk", type=int )
            p.add( "-n", "--n", required=False, help="n most similar items", type=int )

            p.add( "-host", "--host", required=False, help="db host", type=str )
            p.add( "-user", "--user", required=False, help="db user", type=str )
            p.add( "-pwd", "--password", required=False, help="db password", type=str )
            p.add( "-db_name", "--database", required=False, help="db name", type=str )

            p.add( "-owl", "--path_to_owl", required=False, help="path to owl ontology", type=str )
            p.add( "-db_onto", "--path_to_ontology_db", required=False, help="path to ontology db", type=str )

            p.add( "-sim_metric", "--similarity_metric", required=False, help="similarity metric acronym db", type=str )

            p.add( "-prefix", "--items_prefix", required=False, help="items prefix", type=str )
            p.add( "-n_split", "--n_split_dataset", required=False, help="number to split the list of entities",
                   type=int )

            MyConfiguration( p.parse_args() )

        return MyConfiguration.__instance

    def __init__(self, options):

        """
        Virtually private constructor.
        """
        if MyConfiguration.__instance is not None:
            raise Exception( "This class is a singleton!" )
        else:
            self.dataset = options.path_to_dataset

            self.cv = options.cv
            self.topk = options.topk
            self.n = options.n

            self.host = options.host
            self.user = options.user
            self.password = options.password
            self.database = options.database

            self.path_to_owl = options.path_to_owl
            self.path_to_ontology = options.path_to_ontology_db

            self.sim_metric = options.similarity_metric

            self.n_split = options.n_split_dataset
            self.item_prefix = options.items_prefix

        MyConfiguration.__instance = self
