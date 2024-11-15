from elasticsearch import Elasticsearch, helpers, RequestsHttpConnection
import json

class BM25Retriever:
    def __init__(self, es_host, es_port):
        self.es_conn = Elasticsearch([{"host": es_host, "port": es_port}], timeout=50)

    def retrieval_candidates(
            self,
            query,
            top_k=10
    ):
        should_clauses = [{
            'match': {
                "segment": {
                    "query": query,
                }
            }
        }]
        query = {
            'query': {
                'bool': {
                    'should': should_clauses
                }
            },
            "from": 0,
            "size": top_k,
        }
        res = self.es_conn.search(index="ms_marco_21", body=json.dumps(query))
        candidates = []
        for hit in res['hits']['hits']:
            source = hit['_source']
            candidates.append({
                'es_score': hit['_score'],
                **source,
            })
        return candidates
