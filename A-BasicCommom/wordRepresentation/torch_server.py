import argparse
import logging
import sys
import time
from types import SimpleNamespace

import falcon
from falcon_cors import CORS
import json
import waitress
import lawa
from gensim.models import KeyedVectors

if sys.hexversion < 0x03070000:
    ft = time.process_time
else:
    ft = time.process_time_ns
    
logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['*'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')
parser.add_argument(
    '-p', '--port', default=58085,
    help='falcon server port')
parser.add_argument(
    '-m', '--for_search', default=1,
    help='falcon server port')
args = parser.parse_args()
model_config = args.config_file


class TorchResource:

    def __init__(self):
        logger.info("...")
        with open(model_config) as fin:
            config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
        self.model = KeyedVectors.load_word2vec_format(config.trained_weight, binary=False, encoding='utf-8')
        logger.info("###")

    def process_context(self, pos, neg, topn= 1):
        start = ft()
        analog = self.model.most_similar(positive=pos, negative=neg, topn=topn)
        logger.info("cut:{}ns".format(ft() - start))
        return {'data': analog}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        pos = req.get_param('pos', True)
        pos = pos.split(',')
        neg = req.get_param('neg', True)
        neg = neg.split(',')
        resp.media = self.process_context(pos, neg)
        logger.info("###")

    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        start = ft()
        jsondata = json.loads(req.stream.read(req.content_length))
        pos= jsondata['pos']
        neg = jsondata.get('neg', None)
        topn = jsondata.get('topn', 1)
        resp.media = self.process_context(pos, neg, topn)
        logger.info("tot:{}ns".format(ft() - start))
        logger.info("###")


if __name__ == "__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')
