# Bayesian NN via martingale sampling

Let for $i=1,...,N$
$$
x_i \overset{\text{iid}}{\sim} f(\cdot|\theta)
$$
for unknown $\theta$. First set $\theta_0$ and then estimate the parameter as
$$
\theta_i = \theta_{i-1} + \gamma_i \nabla_{\theta}\log f(x_i|\theta_{i-1})
$$
for $i=1,...,N$. Next, iteratively generate new data and update the parameter
$$
\begin{aligned}
x_i &\sim f(\cdot|\theta_{i-1}) \\
\theta_i &= \theta_{i-1} + \gamma_i \nabla_{\theta}\log f(x_i|\theta_{i-1})
\end{aligned}
$$
for $i=N+1,...,N+M$. It eventually converges to something (read the paper), which we can use a sample of $\theta$.
