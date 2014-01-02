import os


from autocause.core import featurize_many
from autocause.challenge import read_all

# for debugging
# from boomlet import settings
# settings.PARALLEL.PMAP = False

pairs = read_all("sample")[0]
featurize_many(pairs, "configs")
