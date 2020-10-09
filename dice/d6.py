import numpy as np
import scipy.stats as stats

from dice.model import DirichletCat
from dice.novelty import DiceNoveltyDetector

# example of each possible die observation category (6, 1)
obs_cat = np.asarray([[i for i in range(1, 7)]]).T
# priors
alpha = np.asarray([[20.0] * 6])
prior_dist = stats.dirichlet(alpha[0])
# model
model = DirichletCat(alpha)
# novelty detector
nov_detect = DiceNoveltyDetector(model, obs_cat)
