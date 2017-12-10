import pandas as pd
import numpy as np
import codecs
from unidecode import unidecode
import re
import sys
import csv

try:
    reload(sys)
except NameError:
    pass
try:
    sys.setdefaultencoding('utf-8')
except AttributeError:
    pass

regex_unprintable = re.compile(r'\[\?\]')
regex_newline = re.compile(r'\n')
regex_carriage_return = re.compile(r'\r')
regex_tabs = re.compile(r'\t')

def cleaning_function(x):
    try:
        x = unicode(x)
    except (TypeError, NameError):
        pass
    if type(x) != str:
        x = str(x)
    x = unidecode(x)
    x.encode("ascii")
    x = regex_unprintable.sub('', x)
    x = x.replace('\n', '').replace('\r', '').replace('"', '')

    return (x)

df = pd.read_csv('../inputs/train.tsv', sep = '\t', encoding = 'utf-8', engine = "python")
print (len(df))
for col in ['name', 'category_name', 'brand_name', 'item_description']:
    print (col)
    df[col] = df[col].apply(lambda x: cleaning_function(x))

df.to_csv('../inputs/train_clean.tsv', sep = '\t', index = False)
