# utf-8
import logging
import sys


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("__name__")

        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(module)s:%(asctime)s, %(message)s")
    
    def __call__(self):
       return self.logger
