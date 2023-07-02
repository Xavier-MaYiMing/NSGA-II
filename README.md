### NSGA-II

##### Reference: Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197.

Nondominated sorting genetic algorithm II (NSGA-II) with simulated binary crossover (SBX) and polynomial mutation.

| Variables   | Meaning                                                      |
| ----------- | ------------------------------------------------------------ |
| npop        | Population size                                              |
| iter        | Iteration number                                             |
| lb          | Lower bound                                                  |
| ub          | Upper bound                                                  |
| pc          | Crossover probability (default = 1)                          |
| eta_c       | The spread factor distribution index (default = 20)          |
| pm          | Mutation probability (default = 0.1)                         |
| eta_m       | The perturbance factor distribution index (default = 20)     |
| dim         | Dimension                                                    |
| pop         | Population                                                   |
| objs        | Objectives                                                   |
| pfs         | pfs[i] means the Pareto front which the i-th individual belongs to |
| rank        | The Pareto rank of all the individuals in the population     |
| cd          | Crowding distance                                            |
| mating_pool | Mating pool                                                  |
| pf          | The obtained Pareto front                                    |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 300, np.array([0] * 10), np.array([1] * 10))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/NSGA-II/blob/main/Pareto%20front.png)

