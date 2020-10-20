import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial


def vis_log_marginal(fair_model, cheat_model, die_dirichlet, threshold=0., title=""):
    """Visualize log marginals of random sample of dice with a scatter plot

    Args:
        fair_model (DirichletCat): Model of a fair die.
        cheat_model (DirichletCat): Model of an unfair die.
        die_dirichlet (scipy.stats.dirichlet): Dirichlet to sample random dice from.
        threshold (float, optional): Threshold for a log Bayes factor comparison. Defaults to 0.
        title (str, optional): Title for the plot. Defaults to "".
    """
    # define constants
    sample_num = 1000
    trial_num = 10
    roll_num = 100
    # sample dice from the Dirichlet
    die_distributions = die_dirichlet.rvs(size=sample_num)
    # get marginals from trials of sets of die rolls of each die
    fair_marginals = []
    cheat_marginals = []
    for i in range(sample_num):
        obs_count = multinomial.rvs(roll_num, die_distributions[i], size=trial_num)
        for j in range(trial_num):
            fair_marginals += [fair_model.log_marginal(obs_count[j])]
            cheat_marginals += [cheat_model.log_marginal(obs_count[j])]
    # scatter plot of the marginals
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(fair_marginals, cheat_marginals, color=[0, 0, 1, .1])
    # add labels
    fig.suptitle(title)
    ax.set_xlabel("Fair Model Log Marginal")
    ax.set_ylabel("Unfair Model Log Marginal")
    # add a line for x = y
    min_val = min(min(fair_marginals), min(cheat_marginals))
    max_val = max(max(fair_marginals), max(cheat_marginals))
    line = np.asarray([min_val, max_val])
    plt.plot(line, line, color="black")
    # draw threshold for a log Bayes factor comparison if given
    if threshold > 0.:
        log_thresh = np.log(threshold)
        plt.plot(line, line + log_thresh, color="red")
        plt.plot(line, line - log_thresh, color="green")
    # show the visualization
    plt.show()
