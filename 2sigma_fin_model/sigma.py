# Here's an example of loading the CSV using Pandas's built-in HDF5 support:
import pandas as pd

with pd.HDFStore("train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
    df.to_csv('train.csv')