from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from glob import glob
import json

from generator.trainer.train_Mistral import config
from .db_handler import DBHandler

class ESHandler(DBHandler):

    def __init__(self, host, port=None, verbose=False):
        super().__init__()
        self.es_conn = self.connect(host=host, port=port)
        self.verbose = verbose

    def connect(self, host, port=None):
        es_conn = Elasticsearch(host=host, port=port)
        if not es_conn.ping():
            raise "Cannot connect to ES"
        if self.verbose:
            print("Connected to Elasticsearch")
        return es_conn

    def create_db(self, index_name, config_path):
        success = False
        settings = json.load(open(config_path))
        try:
            if not self.es_conn.indices.exists(index_name):
                self.es_conn.indices.create(
                    index=index_name,
                    # ignore=400,
                    body=settings,
                )
                success = True
            if self.verbose:
                print("Index created successfully")
        except Exception as e:
            print("Error while creating index: ", str(e))
        finally:
            return success


    def add_records(self, records, index_name):
        pass

    def delete_item(self):
        pass

    def bulk_index(self, index_name, records, batch_size=100):
        """

        @param index_name:
        @param records:
        @param batch_size:
        @return:
        """
        try:
            response = helpers.bulk(
                self.es_conn,
                records,
                index=index_name,
                request_timeout=300,
            )
            if self.verbose:
                print(response)
        except Exception as e:
            print("Error: ", str(e))
