from flask import Flask, cli

import multiprocessing
import os
from flask_cors import CORS

def start():
    cli.show_server_banner = lambda *_: None
    NUM_THREADS = str(multiprocessing.cpu_count())

    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    os.environ["OMP_NUM_THREADS"] = NUM_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = NUM_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
    if os.environ.get("CACHE_DIR"):
        os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]
    

