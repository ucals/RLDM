Introduction
============

Temporal-difference (TD) learning is a core learning technique in modern
reinforcement learning (Sutton, 1988; Kaelbling, Littman & Moore; Sutton
& Barto, 1998; Szepesvári, 2014). TD learning generates good estimates
for expected returns by quickly bootstrapping from other expected-return
estimates. TD($\lambda$) is one of the most popular algorithms in TD
learning. It was first introduced in Sutton’s article *Learning to
Predict by the Methods of Temporal Differences*, one of the most
referenced articles in the field with over 5,000 citations. The purpose
of this paper is to discuss the results obtained by replicating Sutton’s
TD($\lambda$) algorithm, providing a complete description of the
experiments, implementation details and outcomes.



Random Walk
-----------

Sutton (1988) illustrates TD methods with a random-walk example, one of
the simplest dynamic systems. A bounded random walk is a Markov Reward
Process. The episodes always start in the middle state $D$. At each time
step, the walk moves to the neighbor state, either left or right, with
equal probability. The episode ends when walk reaches either extreme
states $A$ or $G$. If termination happens by reaching the extreme right
($G$), environment provides a reward of +1; all the other rewards are
zero.

![image](./images/random_walk.eps) [fig:1]

A typical episode terminating in the extreme right might be the
following state-reward sequence: $D, 0, C, 0, D, 0, E, 0, F, 0, G, 1$.
An example of a walk ending in the extreme left might be:
$D, 0, C, 0, B, 0, A, 0$. This MRP is undiscounted ($\gamma = 1$);
therefore, the true value of each state is the probability of
terminating on the extreme right by starting from that specific state.
Thus, the true values from all states, from $A$ to $G$ respectively, are

$$v_{true} = \Bigg[0, \frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}, 1\Bigg]
    \centering$$

We applied $TD(\lambda)$ algorithm to effectively predict these true
values, replicating Sutton’s original paper methods and results. The
following sections explain in details how we implemented the
experiments, their results, and compares them with the previously
published outcomes.

Results
=======

![image](./images/figure.png) [fig:fig2]

Figure 2 summarizes the results obtained by replicating Sutton’s
experiments. In the left, the chart shows the results of the first
experiment, where RMS errors were calculated between the state-value
function true values and the ones estimated after repeated presentations
until convergence, for different values of $\lambda$. We averaged the
errors over 100 training sets, exactly as done by Sutton. This chart
replicates Figure 3 in the original article.

In the center, the chart shows average errors obtained in a single run
of each training set, averaged over 100 training sets, for different
values of $\alpha$ and $\lambda$ This chart replicates Figure 4 in the
original article.

Finally, in the right, we can see the chart that replicates Figure 5 in
Sutton’s article. The results were produced by experiment 2, after a
single run of each training set, also averaged over 100 training sets,
for different values of $\alpha$ and $\lambda$. For each value of
$\lambda$, we selected the best value of $\alpha$.

As we can see, our results matched the overall results obtained by
Sutton.