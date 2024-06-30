#!/usr/bin/env python
# coding: utf-8

# # Checking analytic bounds' validity and constraints' correctness in Zero-inflated MCAR and MAR model
# 
# In the paper "Zero Inflation as a Missing Data Problem: a Proxy-based Approach", we prove that:
# 1. For zero-inflation MCAR case $X^{(1)} \rightarrow X \leftarrow R \rightarrow W$:
#    1. (**C1**) $p(w_0|r_1) = p(w_0|x_1)$
#    2. (**C2a**) The bound for $p(w_0|x_0)$ is  
#       1. If $OR > 1$ then $\max_x p(w_0 \mid x) < p(w_0 \mid r_0) \leq 1$.  
#       2. If $OR < 1$ then $0 \leq p(w_0 \mid r_0) < \min_x p(w_0 \mid x)$.
#    3. (**C2b**) For any $q(w_0|r_0)$ inside the bound, the matrix inversion equation $D = [\mathbf{q}_{W|R}]^{-1} \mathbf{p}_{WX}$ creates a random matrix $D$.
# 2. In zero-inflation MAR case $R \leftarrow C \rightarrow X^{(1)} \rightarrow X \leftarrow R \rightarrow W$:
#    1. (**M3**) Either $OR(c) > 1, \forall c$ or $OR(c) < 1, \forall c$.
#    2. (**C3**) $p(w_0|w_1) = p(w_0|x_1, c), \forall c$. This leads to a marginal constraint
#       1. (**M5**) $p(w_0|x_1, c) = p(w_0|x_1), \forall c$.
#    3. (**C4a**) The bound for $p(w_0|x_0)$ is  
#       1. If $OR(c) > 1$ then $\max_{x, c} p(w_0 \mid x, c) < p(w_0 \mid r_0) \leq 1$.  
#       2. If $OR(c) < 1$ then $0 \leq p(w_0 \mid r_0) < \min_{x, c} p(w_0 \mid x, c)$.  
#    4. (**C4b**) For any $q(w_0|r_0)$ inside the bound, the matrix inversion equation $D = [\mathbf{q}_{W|R}]^{-1} \mathbf{p}_{WXC}$ creates a random matrix $D$.
#   
# Here $OR = \frac{p(w_1 \mid x_1)}{p(w_0 \mid x_1)} \frac{p(w_0 \mid x_0)}{p(w_1 \mid x_0)}$, and $OR(c) = \frac{p(w_1 \mid x_1, c)}{p(w_0 \mid x_1, c)} \frac{p(w_0 \mid x_0, c)}{p(w_1 \mid x_0, c)}$.
# 
# This code checks these results by simulating random DGPs and compute truth value of those claims.

# In[ ]:


import io
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from pathlib import Path

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete.CPD import TabularCPD


# In[ ]:


# pgmpy throws a warning when method `add_cpds` is used, so disable it
# https://pgmpy.org/_modules/pgmpy/models/BayesianNetwork.html#BayesianNetwork.add_cpds
import logging
logger = logging.getLogger('pgmpy')
logger.setLevel(level=logging.ERROR)


# In[ ]:


# EXP 1
CASE = 'MCAR'     # choose either 'MCAR' or 'MAR'
N = 1000000
seed = 42        # random seed
compute_num_bound = False


# In[ ]:


# EXP 2
CASE = 'MCAR'     # choose either 'MCAR' or 'MAR'
N = 20
seed = 42        # random seed
compute_num_bound = True


# ## MCAR case: $X^{(1)} \rightarrow X \leftarrow R \rightarrow W$

# In[ ]:


def add_ZI_consistency_edge(net):
    card_R = net.get_cardinality("R")
    card_X1 = net.get_cardinality("X1")
    """
    cpd table for p(x | x(1), r)
    ---
    R=   |     0     |     1     |
    X1=  |   0 |   1 |   0 |   1 |
    ---
    X=0  | 1.0 | 1.0 | 1.0 | 0.0 |
    X=1  | 0.0 | 0.0 | 0.0 | 1.0 |
    ---
    """
    cpd_X = TabularCPD(
        "X",
        2,
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        evidence=["R", "X1"],
        evidence_card=[card_R, card_X1],
    )
    net.add_cpds(cpd_X)
    return net


def odd_ratio(p, c=None):
    """
    Input:
        If c=None, p is 2x2 matrix with p[i,j] = p(a=i|b=j).
        If c=0,1, p is 2x2x2 matrix with p[i,j,k] = p(a=i|b=j,c=k).
    Output: Odd-ratio p(a=1|b=1,c) / p(a=0|b=1,c) * p(a=0|b=0,c) / p(a=1|b=0,c)
    """
    if c == None:
        return p[1, 1] / p[0, 1] * p[0, 0] / p[1, 0]
    elif isinstance(c, int):
        return p[1, 1, c] / p[0, 1, c] * p[0, 0, c] / p[1, 0, c]


class CaseMCAR:
    def __init__(self, cards):
        """
        Create the graph X(1) -> X <- R -> W
        """
        self.vertices = ["X1", "X", "R", "W"]
        self.cards = cards
        self.edges = [("X1", "X"), ("R", "X"), ("R", "W")]
        self.net = BayesianNetwork(self.edges)
        # self.net.to_daft(node_pos={'X': (0,0), 'X1': (0,1), 'R': (1,0), 'W': (2,0)}).render()
        self.get_random_cpds()
        # If |x| < epsilon, then x is consider "zero" (small noise)
        self.eps = 1e-12
        self.eps2 = 1e-7
        self.lb, self.ub = None, None
        self.num_lb, self.num_ub = None, None

    def get_random_cpds(self):
        """
        Get a random data-generation process in the model, i.e., satisfying 3 conditions
        1. random cpds
        2. applying the ZI consistency
        3. W must associate to R
        """
        while True:
            self.net.get_random_cpds(n_states=self.cards, inplace=True)
            self.net = add_ZI_consistency_edge(self.net)
            infer = VariableElimination(self.net)
            self.p_WR = infer.query(["W", "R"]).values
            self.p_WX = infer.query(["W", "X"]).values
            # Proxy assumption: W must assoc with X, e.g., OR cannot be 1
            # Marginal assumption: W must assoc with X
            if (odd_ratio(self.p_WR, c=None) != 1) and (
                odd_ratio(self.p_WX, c=None) != 1
            ):
                break

        # Compute the conditional distributions p(w|r) and p(w|x)
        self.p_W_X = np.zeros_like(self.p_WX)
        for x in range(self.cards["X"]):
            self.p_W_X[:, x] = self.p_WX[:, x] / np.sum(self.p_WX[:, x])
        self.p_W_R = np.zeros_like(self.p_WR)
        for r in range(self.cards["R"]):
            self.p_W_R[:, r] = self.p_WR[:, r] / np.sum(self.p_WR[:, r])

    def is_C1(self):
        """
        This function check truth value of result:
        (C1) $p(w_0|r_1) = p(w_0|x_1)$
        Output: Boolean
        """
        ans = np.abs(self.p_W_R[0, 1] - self.p_W_X[0, 1]) < self.eps
        return ans

    def is_C2_bound_valid(self):
        """
        This function check truth value of result (C2):
            a) If $OR > 1$ then $\max_x p(w_0 \mid x) < p(w_0 \mid r_0) \leq 1$.  
            b) If $OR < 1$ then $0 \leq p(w_0 \mid r_0) < \min_x p(w_0 \mid x)$.
            where p(w_0 \mid r_0) is the ground-truth proxy-indicator c.d.f.
        - The bound is valid when OR(a, b) = True
        - The bound is incorrect when OR(a, b) = False
        - AND(a, b) is always False.

        Output: Boolean value of OR(a, b)
        """
        OR = odd_ratio(self.p_WX, c=None)
        if OR > 1:
            self.lb = np.max(self.p_W_X[0, :])
            self.ub = 1
        elif OR < 1:
            self.lb = 0
            self.ub = np.min(self.p_W_X[0, :])
        else:  # This cannot happen as we ruled out this case when we generate DGP
            print("Error: OR == 1!")
            return 2  # error code
        return (self.lb < self.p_W_R[0, 0]) and (self.p_W_R[0, 0] < self.ub)

    def id_p_w0_r1(self):
        """
        p(w0|r1) = p(w0|x1) is identified
        """
        self.hat_p_w0_r1 = self.p_WX[0, 1] / np.sum(self.p_WX[:, 1], axis=0)
        return self.hat_p_w0_r1

    def is_C2_bound_int(self, n):
        """
        Is bound interior valid?
        1. Selecting n values of p(w0|r0) inside the calculated bounds.
        2. Creating p(W|R) and check if p(RX) = [p(W|R)]^{-1} p(WX) is a random matrix.
        """
        assert self.lb < self.ub

        self.id_p_w0_r1()
        p_RXs = []
        ps = np.linspace(self.lb + 1e-5, self.ub - 1e-5, n)
        for p in ps:
            p_W_R = np.asarray(
                [
                    [p, self.hat_p_w0_r1],
                    [1 - p, 1 - self.hat_p_w0_r1],
                ]
            )
            p_RX = np.linalg.inv(p_W_R) @ self.p_WX.reshape(2, -1)
            p_RXs.append(p_RX.flatten())
        p_RXs = np.asarray(p_RXs)
        is_nonnegative = (p_RXs >= -self.eps2).all()  # can be a small negative noise
        is_addingto1 = (np.max(np.abs(1 - np.sum(p_RXs, axis=1))) <= self.eps2).all()
        return is_nonnegative and is_addingto1

    def get_numerical_bound(self, seed:int, dgp:int):
        """
        Computing numerical bound using autobound package from
        the supplement of Duarte et al. (2023) (https://doi.org/10.1080/01621459.2023.2216909).
        Newer version is from this Docker: docker run -p 8888:8888 -it gjardimduarte/autolab:v4
        """
        from autobounds.causalProblem import causalProblem
        from autobounds.DAG import DAG

        # Create the graph
        # the ZI counterfactual X(1) is denoted by Y1, as `autobound` cannot handle similar variable names yet
        dag = DAG()
        dag.from_structure("Y1 -> X, R -> X, R -> W")
        problem = causalProblem(dag)

        # This works better
        # Adding Zi consistency constraint: p(X=1 | R=0, X(1)=0) = 0 => p(X=1,R=0,other_vars) = 0
        """probability table for MCAR
        W,Y1, X, R, prob
        0, 0, 1, 0, 0.0
        0, 1, 1, 0, 0.0
        1, 0, 1, 0, 0.0
        1, 1, 1, 0, 0.0
        """
        cartesian_prod_WX1 = np.vstack(
            list(product(range(self.cards["W"]), range(self.cards["X1"])))
        )
        n = len(cartesian_prod_WX1)
        data_ZI_0 = pd.DataFrame(  # probability table
            np.hstack(
                (
                    cartesian_prod_WX1,
                    np.ones((n, 1)),
                    np.zeros((n, 1)),
                    np.zeros((n, 1)),
                )
            ),
            columns=["W", "Y1", "X", "R", "prob"],
        )
        data_ZI_0 = io.StringIO(data_ZI_0.to_csv(index=False))
        problem.load_data(data_ZI_0, optimize=True)
        # Adding Zi consistency constraint: p(X=x | R=1, X(1)=x) = 1 => p(X=1,R=1,X1=0, other_vars) = 0
        data_ZI_1 = pd.DataFrame(  # probability table
            [[1, 1, 0, 0, 0.0], [1, 1, 0, 1, 0.0]],
            columns=["X", "R", "Y1", "W", "prob"],
        )
        data_ZI_1 = io.StringIO(data_ZI_1.to_csv(index=False))
        problem.load_data(data_ZI_1, optimize=True)

        # Axioms of probability constraints
        problem.add_prob_constraints()  # sum to 1
        for para in problem.parameters:  # non-negative
            problem.add_constraint([(para[0], [para[1]])], symbol=">=")

        # Adding observational data
        data_p_WX = io.StringIO(
            pd.DataFrame(
                {"W": [0, 0, 1, 1], "X": [0, 1, 0, 1], "prob": self.p_WX.flatten()}
            ).to_csv(index=False)
        )
        problem.load_data(data_p_WX, optimize=True)

        # Adding estimands - p(w=0|r=0) = p(W(R=0)=0)
        problem.set_estimand(problem.query("W(R=0)=0"))

        # Writing optimization programs
        prog = problem.write_program()

        # Writing problem file to solve directly with scip in terminal
        for sense in ['max','min']:
            Path(f"./dgp{dgp}").mkdir(parents=True, exist_ok=True)
            prog.to_pip(f"./dgp{dgp}/seed{seed}_dgp{dgp}_{sense}.pip", sense=sense)
        return


# In[ ]:


if CASE == "MCAR":
    print(f"MCAR case. Graph: X(1) -> X <- R -> W")
    cards = {"X1": 2, "X": 2, "R": 2, "W": 2}
    case_mcar = CaseMCAR(cards)

    np.random.seed(seed)
    if compute_num_bound:
        """
        Experiment 2: BOUND CORROBORATION VIA NUMERICAL METHODS
        1. Compute analytical bounds for N DGPs
        2. Write polynomial program to be solved for numerical bounds
        3. You should compare the bounds manually by yourself
        """
        N = 50
        file_name = f"mcar_compare-seed{seed}-N{N}.csv"
        results = []
        for dgp in tqdm(range(N)):
            case_mcar.get_random_cpds()
            # compute analytic bounds
            _ = case_mcar.is_C2_bound_valid()
            # write program to solve for num bounds using scip directly in terminal
            _ = case_mcar.get_numerical_bound(seed, dgp)
            results.append(
                    [
                        seed,
                        dgp,
                        case_mcar.lb,
                        case_mcar.ub,
                        None,
                        None,
                        case_mcar.p_W_R[0, 0]
                    ]
                )
        # save results
        pd.DataFrame(
            np.vstack(results, dtype=object),
            columns=['seed','dgp','lb','ub','num_lb','num_ub','p_w0_r0']
        ).to_csv(file_name, mode='w', index=False)
    else:
        """
        Experiment 1: BOUND VALIDITY
        1. Check claims of ZI MCAR theorem, here denoted as C1, C2a, C2b
        """
        file_name = f"mcar_results-seed{seed}-N{int(N/1000)}k.txt"
        K = np.min([200, N])    # Save file every K steps
        columns = [
            "seed",
            "DGP_#",
            "is_C1",
            "is_C2_vlid",
            "is_C2_bint",
            "lb",
            "ub",
            *[f"p_w0_r{i}" for i in range(cards['R'])],
            *[f"p_w0_x{i}" for i in range(cards['X'])],
        ]
        with open(file_name, 'w') as f:
            f.write(','.join(columns)+'\n')
            f.close
        for b in tqdm(range(int(N / K))):
            results = []
            for i in range(K):
                dgp = b*K + i
                case_mcar.get_random_cpds()

                re_C1 = case_mcar.is_C1()
                re_C2_vlid = case_mcar.is_C2_bound_valid()
                case_mcar.id_p_w0_r1()
                re_C2_bint = case_mcar.is_C2_bound_int(20)
    
                results.append(
                    [
                        seed,
                        dgp,
                        re_C1,
                        re_C2_vlid,
                        re_C2_bint,
                        case_mcar.lb,
                        case_mcar.ub,
                        *case_mcar.p_W_R[0, :],
                        *case_mcar.p_W_X[0, :],
                    ]
                )
            # save results
            pd.DataFrame(
                np.vstack(results, dtype=object), columns=columns
            ).to_csv(file_name, mode='a', index=False, header=False)

        results = pd.read_csv(file_name)
        re_C1 = np.sum(results["is_C1"]).astype(int)
        re_C2_vlid = np.sum(results["is_C2_vlid"]).astype(int)
        re_C2_bint = np.sum(results["is_C2_bint"]).astype(int)
        print(
            f"(C1) p(w0|r1) is identified and equals p(w0|x1) for {re_C1}/{N} times ({int(re_C1*100/N)}%)."
        )
        print(
            f"(C2_v) The bound is valid for {re_C2_vlid}/{N} times ({int(re_C2_vlid*100/N)}%)."
        )
        print(
            f"(C2_i) The bound interior is valid for {re_C2_bint}/{N} times ({int(re_C2_bint*100/N)}%)."
        )


# ## MAR case: $R \leftarrow C \rightarrow X^{(1)} \rightarrow X \leftarrow R \rightarrow W$

# In[ ]:


class CaseMAR:
    def __init__(self, cards):
        """
        Create the graph R <- C -> X(1) -> X <- R -> W
        """
        self.vertices = ["X1", "X", "R", "W", "C"]
        self.cards = cards
        print(self.cards)
        self.edges = [("X1", "X"), ("R", "X"), ("R", "W"), ("C", "X1"), ("C", "R")]
        self.net = BayesianNetwork(self.edges)
        # self.net.to_daft(node_pos={'X': (0,0), 'X1': (0,1), 'R': (1,0), 'W': (2,0), 'C': (1,1)}).render()
        self.get_random_cpds()
        # If |x| < epsilon, then x is consider "zero" (small noise)
        self.eps = 1e-12
        self.eps2 = 1e-7
        self.lb, self.ub = None, None
        self.num_lb, self.num_ub = None, None

    def get_random_cpds(self):
        """
        Get a random data-generation process in the model, i.e., satisfying 3 conditions
        1. random cpds
        2. applying the ZI consistency
        3. W must associate to R
        """
        while True:
            self.net.get_random_cpds(n_states=self.cards, inplace=True)
            self.net = add_ZI_consistency_edge(self.net)
            infer = VariableElimination(self.net)
            self.p_WR = infer.query(["W", "R"]).values
            self.p_WXC = infer.query(["W", "X", "C"]).values
            # Proxy assumption: W must assoc with X, e.g., OR cannot be 1
            # Marginal assumption: W must assoc with X conditional on C
            conditions = [odd_ratio(self.p_WXC, c) != 1 for c in range(self.cards["C"])]
            if odd_ratio(self.p_WR, c=None) != 1 and np.all(conditions):
                break

        # Compute the conditional distributions p(w|r) and p(w|x,c)
        self.p_W_XC = np.zeros_like(self.p_WXC)
        for c in range(self.cards["C"]):
            for x in range(self.cards["X"]):
                self.p_W_XC[:, x, c] = self.p_WXC[:, x, c] / np.sum(self.p_WXC[:, x, c])
        self.p_W_R = np.zeros_like(self.p_WR)
        for r in range(self.cards["R"]):
            self.p_W_R[:, r] = self.p_WR[:, r] / np.sum(self.p_WR[:, r])

    def is_C3M5(self):
        """
        Check truth value of C3 and M5
        - "C3 and M5 are true" when p(w0 | r1) =  p(w0 | x1, c) for all c
        - False otherwise.
        Input: BayesianNet of this case
        output: Boolean value of AND(C3, M5)
        """
        max_val = np.max([*self.p_W_XC[0, 1, :], self.p_W_R[0, 1]])
        min_val = np.min([*self.p_W_XC[0, 1, :], self.p_W_R[0, 1]])
        return (max_val - min_val) < self.eps

    def is_M3(self):
        """
        Check truth value of result M3.
        Output: Boolean value of M3.
        """
        # OR could be equal 1, so need to check both
        is_M2_smaller_1 = [odd_ratio(self.p_WXC, c) < 1 for c in range(self.cards["C"])]
        is_M2_smaller_1 = np.all(is_M2_smaller_1)
        is_M2_larger_1 = [odd_ratio(self.p_WXC, c) > 1 for c in range(self.cards["C"])]
        is_M2_larger_1 = np.all(is_M2_larger_1)
        return is_M2_smaller_1 or is_M2_larger_1

    def is_C4_bound_valid(self):
        """
        This function check truth value of result (C4):
            a) If $OR(c) > 1$ then $\max_{x,c} p(w_0 \mid x, c) < p(w_0 \mid r_0) \leq 1$.  
            b) If $OR(c) < 1$ then $0 \leq p(w_0 \mid r_0) < \min_{x,c} p(w_0 \mid x, c)$.
            where p(w_0 \mid r_0) is the ground-truth proxy-indicator c.d.f.
        - The bound is valid when OR(a, b) = True
        - The bound is incorrect when OR(a, b) = False
        - AND(a, b) is always False.

        Output: Boolean value of OR(a, b)
        """
        OR = odd_ratio(self.p_WXC, c=1)  # Assuming M2 is correct
        if OR > 1:
            self.lb = np.max(self.p_W_XC[0, :, :])
            self.ub = 1
        elif OR < 1:
            self.lb = 0
            self.ub = np.min(self.p_W_XC[0, :, :])
        else:  # This cannot happen as we ruled out this case when we generate DGP
            print("Error: OR == 1!")
            return 2  # error code
        return (self.lb < self.p_W_R[0, 0]) and (self.p_W_R[0, 0] < self.ub)

    def id_p_w0_r1(self):
        """
        p(w0|r1) = p(w0|x1) is identified
        """
        self.hat_p_w0_r1 = self.p_WXC[0, 1, :] / np.sum(self.p_WXC[:, 1, :], axis=0)
        self.hat_p_w0_r1 = self.hat_p_w0_r1.mean()
        return self.hat_p_w0_r1

    def is_C4_bound_int(self, n):
        """
        Is the bound interior valid?
        1. Selecting n values of p(w0|r0) inside the calculated bounds.
        2. Creating p(W|R) and check if p(RX) = [p(W|R)]^{-1} p(WXC) is a random matrix.
        """
        assert self.lb < self.ub

        self.id_p_w0_r1()
        p_RXCs = []
        ps = np.linspace(self.lb + 1e-5, self.ub - 1e-5, n)
        for p in ps:
            p_W_R = np.asarray(
                [
                    [p, self.hat_p_w0_r1],
                    [1 - p, 1 - self.hat_p_w0_r1],
                ]
            )
            p_RXC = np.linalg.inv(p_W_R) @ self.p_WXC.reshape(2, -1)
            p_RXCs.append(p_RXC.flatten())
        p_RXCs = np.asarray(p_RXCs)
        is_nonnegative = (p_RXCs >= -self.eps2).all()  # can be a small negative noise
        is_addingto1 = (np.max(np.abs(1 - np.sum(p_RXCs, axis=1))) <= self.eps2).all()
        return is_nonnegative and is_addingto1

    def get_numerical_bound(self, seed:int, dgp:int):
        """
        Computing numerical bound using autobound package from
        the supplement of Duarte et al. (2023) (https://doi.org/10.1080/01621459.2023.2216909).
        Newer version is from this Docker: docker run -p 8888:8888 -it gjardimduarte/autolab:v4
        """
        from autobounds.causalProblem import causalProblem
        from autobounds.DAG import DAG

        # Create the graph
        # the ZI counterfactual X(1) is denoted by Y1, as `autobound` cannot handle similar variable names yet
        dag = DAG()
        dag.from_structure("C -> Y1, C -> R, Y1 -> X, R -> X, R -> W")
        problem = causalProblem(dag)

        # This works better
        # Adding Zi consistency constraint: p(X=1 | R=0, X(1)=0) = 0 => p(X=1,R=0,other_vars) = 0
        """probability table for MCAR
        W,Y1, X, R, prob
        0, 0, 1, 0, 0.0
        0, 1, 1, 0, 0.0
        1, 0, 1, 0, 0.0
        1, 1, 1, 0, 0.0
        """
        cartesian_prod_WX1C = np.vstack(
            list(
                product(
                    range(self.cards["W"]),
                    range(self.cards["X1"]),
                    range(self.cards["C"]),
                )
            )
        )
        n = len(cartesian_prod_WX1C)
        data_ZI_0 = pd.DataFrame(  # probability table
            np.hstack(
                (
                    np.ones((n, 1), dtype=int),
                    np.zeros((n, 1), dtype=int),
                    cartesian_prod_WX1C,
                    np.zeros((n, 1), dtype=float),
                ),
                dtype=object,
            ),
            columns=["X", "R", "W", "Y1", "C", "prob"],
        )
        data_ZI_0 = io.StringIO(data_ZI_0.to_csv(index=False))
        problem.load_data(data_ZI_0, optimize=True)
        # Adding Zi consistency constraint: p(X=x | R=1, X(1)=x) = 1 => p(X=1,R=1,X1=0,other_vars) = 0
        cartesian_prod_WC = np.vstack(
            list(
                product(
                    range(self.cards["W"]),
                    range(self.cards["C"]),
                )
            )
        )
        n = len(cartesian_prod_WC)
        data_ZI_1 = pd.DataFrame(  # p(X=1,R=1,X1=0,other_vars) = 0
            np.hstack(
                (
                    np.ones((n, 1), dtype=int),
                    np.ones((n, 1), dtype=int),
                    np.zeros((n, 1), dtype=int),
                    cartesian_prod_WC,
                    np.zeros((n, 1), dtype=float),
                ),
                dtype=object,
            ),
            columns=["X", "R", "Y1", "W", "C", "prob"],
        )
        problem.load_data(io.StringIO(data_ZI_1.to_csv(index=False)), optimize=True)
        # p(X=0,R=1,X1=1,other_vars) = 0
        data_ZI_1.loc[:, ["X", "Y1"]] = 1 - data_ZI_1.loc[:, ["X", "Y1"]]
        problem.load_data(io.StringIO(data_ZI_1.to_csv(index=False)), optimize=True)

        # Axioms of probability constraints
        problem.add_prob_constraints()  # sum to 1
        for para in problem.parameters:  # non-negative
            problem.add_constraint([(para[0], [para[1]])], symbol=">=")

        # Adding observational data
        cartesian_prod_WXC = np.vstack(
            list(
                product(
                    range(self.cards["W"]),
                    range(self.cards["X"]),
                    range(self.cards["C"]),
                )
            )
        )
        data_p_WXC = pd.DataFrame(
            np.hstack((cartesian_prod_WXC, self.p_WXC.reshape(-1, 1)), dtype=object),
            columns=["W", "X", "C", "prob"],
        )
        data_p_WXC = io.StringIO(data_p_WXC.to_csv(index=False))
        problem.load_data(data_p_WXC, optimize=True)

        # Adding estimands - p(w=0|r=0) = p(W(R=0)=0)
        problem.set_estimand(problem.query("W(R=0)=0"))

        # Writing optimization programs
        prog = problem.write_program()

        # Writing problem file to solve directly with scip in terminal
        for sense in ['max','min']:
            Path(f"./dgp{dgp}").mkdir(parents=True, exist_ok=True)
            prog.to_pip(f"./dgp{dgp}/seed{seed}_dgp{dgp}_{sense}.pip", sense=sense)
        return


# In[ ]:


if CASE == "MAR":
    print(f"MAR case. Graph: R <- C -> X(1) -> X <- R -> W")
    cards = {"X1": 2, "X": 2, "R": 2, "W": 2, "C": 2}
    case_mar = CaseMAR(cards)

    np.random.seed(seed)
    if compute_num_bound:
        """
        Experiment 2: BOUND CORROBORATION VIA NUMERICAL METHODS
        1. Compute analytical bounds for N DGPs
        2. Write polynomial program to be solved for numerical bounds
        3. You should compare the bounds manually by yourself
        """
        N = 50
        file_name = f"mar_compare-seed{seed}-N{N}.csv"
        results = []
        np.random.seed(42)
        for dgp in tqdm(range(N)):
            case_mar.get_random_cpds()
            # compute analytic bounds
            _ = case_mar.is_C4_bound_valid()
            # write program to solve for num bounds using scip directly in terminal
            _ = case_mar.get_numerical_bound(seed, dgp)
            results.append(
                [seed, dgp, case_mar.lb, case_mar.ub, None, None, case_mar.p_W_R[0, 0]]
            )
        # save results
        pd.DataFrame(
            results, columns=["seed", "dgp", "lb", "ub", "num_lb", "num_ub", "p_w0_r0"]
        ).to_csv(file_name, mode="w", index=False)
    else:
        """
        Experiment 1: BOUND VALIDITY
        1. Check claims of ZI MCAR theorem, here denoted as C3, C4a, C4b and constraints M3, M5
        """
        file_name = f"mar_results-seed{seed}-N{int(N/1000)}k.txt"
        K = np.min([200, N])  # Save file every K steps
        columns = [
            "seed",
            "DGP_#",
            "is_C3M5",
            "is_M3",
            "is_C4_vlid",
            "is_C4_bint",
            "lb",
            "ub",
            *[f"p_w0_r{i}" for i in range(cards["R"])],
            *[
                f"p_w0_x{i}c{j}"
                for i, j in product(range(cards["X"]), range(cards["C"]))
            ],
        ]
        with open(file_name, "w") as f:
            f.write(",".join(columns) + "\n")
            f.close
        for b in tqdm(range(int(N / K))):
            results = []
            for i in range(K):
                dgp = b * K + i
                case_mar.get_random_cpds()

                re_C3M5 = case_mar.is_C3M5()
                re_M3 = case_mar.is_M3()
                re_C4_vlid = case_mar.is_C4_bound_valid()
                case_mar.id_p_w0_r1()
                re_C4_bint = case_mar.is_C4_bound_int(20)

                results.append(
                    [
                        seed,
                        i,
                        re_C3M5,
                        re_M3,
                        re_C4_vlid,
                        re_C4_bint,
                        case_mar.lb,
                        case_mar.ub,
                        *case_mar.p_W_R[0, :],
                        *case_mar.p_W_XC[0, :, :].flatten(),
                    ]
                )
            # save results
            pd.DataFrame(results, columns=columns).to_csv(
                file_name, mode="a", index=False, header=False
            )

        results = pd.read_csv(file_name)

        re_C3M5 = np.sum(results["is_C3M5"]).astype(int)
        re_M3 = np.sum(results["is_M3"]).astype(int)
        re_C4_vlid = np.sum(results["is_C4_vlid"]).astype(int)
        re_C4_bint = np.sum(results["is_C4_bint"]).astype(int)
        print(
            f"(C3,M5) p(w0|r1) is identified and equals p(w0|x1, c) for all c, for {re_C3M5}/{N} times ({int(re_C3M5*100/N)}%)."
        )
        print(
            f"(M3) The odd-ratio marginal constraint is correct for {re_M3}/{N} times ({int(re_M3*100/N)}%)"
        )
        print(
            f"(C4_v) The bound is valid for {re_C4_vlid}/{N} times ({int(re_C4_vlid*100/N)}%)."
        )
        print(
            f"(C4_i) The bound interior is valid for {re_C4_bint}/{N} times ({int(re_C4_bint*100/N)}%)."
        )

