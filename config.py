import os
import platform

if platform.node() == 'RES-C02RF0T2.local': #example of using a computer specific path
    DATA_DIR = "/Users/lfrench/Downloads/testseate/clp2017_release/data/"
    CORES = 2
else: #using working directory as per setup in the readme file
    cwd = os.getcwd()
    DATA_DIR = os.path.join(cwd, "data")
    CORES = 1

if not (os.path.exists(os.path.join(DATA_DIR, "raw", "clpsych17-test-labels.tsv")) & os.path.exists(
        os.path.join(DATA_DIR, "raw", "clpsych16-data")) & os.path.exists(
        os.path.join(DATA_DIR, "raw", "clpsych17-data")) & os.path.exists(
        os.path.join(DATA_DIR, "raw", "clpsych17-test"))):
    raise RuntimeError(
        'ERROR: this computer (' + platform.node() + ') is not configured. Please change this in config.py')

POSTS_DIR = os.path.join(DATA_DIR, "raw")

interim_folder = os.path.join(DATA_DIR, 'interim')
if not os.path.exists(interim_folder):
    os.makedirs(interim_folder)

