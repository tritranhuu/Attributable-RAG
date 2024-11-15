import argparse
import os
import gzip
import json

from tqdm import tqdm
from glob import glob
from indexer.databases import ESHandler

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index_name", required=True, help="index name", type=str)
parser.add_argument("-d", "--database", required=True, help="database type. Currently support elasticsearch (ES) and MongoDB (Mongo)", type=str)
parser.add_argument("-p", "--path", required=True, help="path to the doc directory", type=str)
parser.add_argument("-c", "--config", required=True, help="path to the config file", type=str)
parser.add_argument("-h", "--host", required=True, help="host name", type=str)
parser.add_argument("--port", required=True, help="port number", type=str)
parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

args = parser.parse_args()

if __name__ == '__main__':
    i = 0
    verbose = args.verbose
    if args.database == "ES":
        db_handler = ESHandler(host=args.host, port=args.port, verbose=verbose)
    else:
        db_handler = None
        assert "Not Implemented"

    data_dir = args.path
    for fn in glob(os.path.join(data_dir, "*.json.gz")):
        if verbose:
            print(f"Indexing {fn}")
        with gzip.open(fn, 'rb') as f:
            batch = []
            bar = tqdm(f)
            for line in bar:
                i += 1
                batch.append(json.loads(line))
                if len(batch) == 5000:
                    db_handler.bulk_index(args.index_name, batch)
                    batch = []
                    bar.set_description(f'Imported {i}')
            if len(batch) > 0:
                db_handler.bulk_index(args.index_name, batch)
                batch = []
                bar.set_description(f'Imported {i}')



