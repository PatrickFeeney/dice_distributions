import numpy as np

from dice.model import DirichletCat
from dice.novelty import DiceNoveltyDetector

# example of each possible die observation category (6, 1)
obs_cat = np.asarray([[i for i in range(1, 7)]]).T
# priors
alpha_fair = np.asarray([[20.0] * 6])
alpha_cheat = np.asarray([[1.0] * 6])
# models
fair_model = DirichletCat(alpha_fair)
cheat_model = DirichletCat(alpha_cheat)
# novelty detector
nov_detect = DiceNoveltyDetector(fair_model, cheat_model, obs_cat)
