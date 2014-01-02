#!/bin/env python

import os


from autocause.core import featurize_many
from autocause.challenge import read_all

# for debugging
# from boomlet import settings
# settings.PARALLEL.PMAP = False

pairs = read_all("final_train")[0]
featurize_many(pairs, "configs")
