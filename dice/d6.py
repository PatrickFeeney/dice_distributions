import numpy as np
from scipy.stats import dirichlet

from dice.model import DirichletCat
from dice.novelty import DiceNoveltyDetector
from dice.visualize import vis_log_marginal

# example of each possible die observation category (6, 1)
obs_cat = np.asarray([[i] for i in range(1, 7)])
# priors
alpha_fair = np.asarray([[20.0] * 6])
alpha_cheat = np.asarray([[1.0] * 6])
# models
fair_model = DirichletCat(alpha_fair)
cheat_model = DirichletCat(alpha_cheat)
# novelty detector
nov_detect = DiceNoveltyDetector(fair_model, cheat_model, obs_cat)


def vis_models():
    vis_log_marginal(fair_model, cheat_model, dirichlet(alpha_fair[0]), threshold=10.0,
                     title="Fair Alpha")
    vis_log_marginal(fair_model, cheat_model, dirichlet(alpha_cheat[0]), threshold=10.0,
                     title="Unfair Alpha")
