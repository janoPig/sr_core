# Algorithm description

***The HROCH parameters***

- $pop_{size}$ Number of individuals in the population (default value is equal to 64).
- $pop_{sel}$ The size of a tournament selection in each iteration (default value is equal to 4).
- $code_{min size}, code_{max size}$ Minimum/maximum allowed size for a individual.
- $const_{size}$ Maximum alloved constants in symbolic model.
- $predefined_{const prob}$ Probability of selecting one of the predefined constants during equations search.
- $predefined_{const set}$ Predefined constants used during equations search.
- $problem$ Set of mathematical functions used in searched equations. Each mathematical function can have a defined weight with which it will be selected in the mutation process. By default, common mathematical operations such as multiplication and addition have a higher weight than goniometric functions or $pow$ and $\exp$. This is a natural way to eliminate $\sin\left(\sin\left(\exp(x)\right)\right)$-type equations, which may have high precision and low complexity, but are usually inappropriate and difficult to interpret.
- $feature_{probs}$ The probability that a mutation process will select a feature. This parameter allows using feature importances provided by a black-box regressor as an input parameter for symbolic regression to speed up the search by selecting mainly the important features.
- $metric$ Metric used to verify goodness of solutions in the search process. Choose from MSE, MAE, MSLE, LogLoss.
- $transformation$ Final transformation for computed value. Choose from logistic function for a classification tasks, no transformation for a regression problems, and ordinal(rounding) for a ordinal regression.
- $sample_{weight}$ Array of weights that are assigned to individual samples.
- $class_{weight}$ Weights associated with classes for a classification tasks with imbalanced classes distribution.

***Stopping criteria***

- $time_{limit}$: Time limit is reached
- $iter_{limit}$: Number of iteration has exceeded.

***Fitness function*** Can be controlled by $metric$ parameter. To verify goodness of solutions for a classification task in the search process is by default used $LogLoss$ function combined with a logistic transformation.

$$
\texttt{LogLoss}(y, p) = -\frac{1}{N} \sum_{i=1}^{N} [ y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) ] \cdot w_i \cdot c[y_i]
$$

For a regression task is used a $\texttt{MSE}$ function.

$$
\texttt{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \cdot w_i
$$

where:

- $N$: Number of examples in the dataset
- $y_i$: Ground truth (true value) for the i-th example
- $\hat{y}_i$: Predicted value for the i-th example
- $p_i$: Predicted probability of the positive class for the i-th example
- $w_i$: Sample weight
- $c[y_i]$: Class weight for a given class $y_i$

***Tournament selection*** Among the current population of solutions (models), the $pop_{sel}$ of them are selected randomly and one with the best fitness score has been returned afterwards.

***Equations representation*** Searched equations are represented as a fixed-length computer program encoded in three-address[*wiki](https://en.wikipedia.org/wiki/Three-address_code) instructions code.

$$
tmp_{i} = op_i(src_{1i}, src_{2i})
$$

If the $\text{op}$ is unary, $\text{src}_{2i}$ is not used. The result of the last operation is taken as the global result. Not all instructions in given program must be used. The three-address code was also used in Multi Expression Programming [*pdf](https://mepx.github.io/oltean_mep.pdf) [*wiki](https://en.wikipedia.org/wiki/Multi_expression_programming)

$src_{1i}$ and $src_{2i}$ can point to one of variables, another instruction with lower index, or to one from constants. Each individual have a fixed number of constants given by algorithm parameter $const_{size}$.

***Mutation*** Each random neighbor generation procedure consists of one code mutation and one constant mutation. The code mutation selects one random instruction from the instructions used and randomly changes their mathematical operation or operand sources, or both. If the $src_{1i}$ or $src_{2i}$ refers to the result of another instruction, then with 50% probability the mutation is performed on that instruction as well. To mutation $op_i$ is used weighted random sampling algorithm which use weights from $problem$ parameter given as input to algorithm. When a result of mutation $src_{1i}$ or $src_{2i}$ is a variable(feature) then is also weighted random sampling algorithm used with weights given in $feature_{probs}$ parameter. Basic constant mutation consist of multiplication or division mutated constant by $\delta$ value.

$$
\delta = 1 + \xi^4 + \epsilon, \quad \xi \sim U(0, 1)
$$

$$
c_{new} = c_{old} * \delta \lor c_{new} = c_{old} / \delta
$$

where:

- $\xi$ A random variable in interval (0, 1)
- $\epsilon$ A very small constant ($10^{-6}$)

If a $predefined_{const set}$ parameter is given then mutation will with $predefined_{const prob}$ probability select one from predefined constants given to algorithm.

***Basic HROCH scheme.*** The algorithm is based on the concept of hill climbing that is suited also to run in a parallel mode. The basic hill climbing algorithm is a simple heuristic search algorithm which belongs to the class of local search–based algorithms. In the basic hill climbing, the algorithm starts with an initial solution and then iteratively makes small changes to improve the solution. The algorithm usually terminates when none of the small changes of the current best solution yields an improvement. Note that the HROCH algorithm, unlike basic hill climbing, works with a population of independent solutions that compete for the time allotted for their evolution (tournament selection). Like Tabu's search, it improves the performance of local search by relaxing its basic rule. At each step, the best candidate replaces the previous solution unless there is a significant worsening of the score. Implementation of Tabu list for the symbolic regression problem can be complicated because the problem consists of a discrete(finding a suitable equation) and a continuous(fine-tuning the constants used) optimization problem. Instead, two contradictory ideas are used. Choosing the best solution from n generated neighbors tends to the solution with the better score. Not following the strict rule that a necessarily better solution must be found avoids getting stuck in a local minimum.

## Pseudocode

**Input:** Input training dataset $D_{tr}$

**Control parameters:** $sample_{size}, pop_{sel}, neighbours_{count}, worsening_{limit}$

**Output:** best symbolic formula solution $bs$

**procedure** Fit($D_{tr}$)

&nbsp;&nbsp;&nbsp;&nbsp; $population \leftarrow RandomIndividuals()$

&nbsp;&nbsp;&nbsp;&nbsp; **for** $individual \in population$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual.D_{tr}^{'} \leftarrow Sample(D_{tr}, sample_{size})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual.score^{'}, individual.D_{tr}^{''} \leftarrow Evaluate(neighbour, neighbour.D_{tr}^{'})$ // $individual.D_{tr}^{''}$ is a small portion of the data (the batch with size 64 rows) from sample $D_{tr}^{'}$ with the worst score.

&nbsp;&nbsp;&nbsp;&nbsp; **while** stoping criteria is not met **do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual \leftarrow tournament(population, pop_{sel})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $bestNeighbour \leftarrow \emptyset$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $i \in (0, neighbours_{count})$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $neighbour \leftarrow individual.currentSolution$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $DoMutation(neighbour)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $neighbour.score^{''} \leftarrow Evaluate(neighbour, individual.D_{tr}^{''})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **if** $neighbour.score^{''} \lt individual.currentSolution.score^{''}*(1 + worsening_{limit})$ **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $neighbour.score^{'}, individual.D_{tr}^{''} \leftarrow Evaluate(neighbour, individual.D_{tr}^{'})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **if** $neighbour.score^{'} \lt bestScore$ **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bestNeighbour = neighbour

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **if** $bestNeighbour \neq \emptyset$ **and** $bestNeighbour.score^{'} \lt individual.bestSolution.score{'} * (1 + worsening_{limit})$ **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual.currentSolution \leftarrow bestNeighbour$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **if** $bestNeighbour.score^{'} \lt individual.bestSolution.score{'}$ **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual.bestSolution \leftarrow bestNeighbour$

&nbsp;&nbsp;&nbsp;&nbsp; **for** $individual \in population$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $individual.score \leftarrow Evaluate(individual, D_{tr})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **if** $individual.score \lt bs.score$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $bs \leftarrow individual$

&nbsp;&nbsp;&nbsp;&nbsp; **return** $bs$

**end procedure**
