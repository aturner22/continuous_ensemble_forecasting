Title
From Deterministic Forecasts to Probabilistic Calibration via Approximate Bayesian Computation

Abstract
A concise formal statement of the problem, methodology, and findings: Given deterministic forecasts lacking uncertainty quantification, we seek to construct calibrated probabilistic post-processors through simulation-based posterior inference, constrained by scoring-rule compliance. We establish an ABC-Gibbs framework for this purpose and assess its statistical and computational efficacy in a high-dimensional meteorological context.

1. Introduction
Define the problem formally: transition from deterministic mappings 
𝑓
:
𝑋
→
𝑋
f:X→X to calibrated conditional distributions 
𝜋
~
𝜃
(
𝑥
𝑡
+
1
∣
𝑥
𝑡
)
π
~
  
θ
​
 (x 
t+1
​
 ∣x 
t
​
 ).

Motivating application: numerical weather prediction (NWP), operational constraints, and epistemic limitations of deterministic forecasts.

Overview of the thesis structure and contributions.

2. Forecasts, Uncertainty, and Calibration
Formalize forecast quality: reliability, resolution, and uncertainty as constraints.

Define proper scoring rules 
𝑆
:
𝑃
(
𝑋
)
×
𝑋
→
𝑅
S:P(X)×X→R and their empirical aggregates 
𝑆
^
(
𝜋
𝜃
,
𝐷
)
S
^
 (π 
θ
​
 ,D).

Define the constrained parameter set:

𝐶
𝜖
=
{
𝜃
∈
Θ
:
𝑆
^
(
𝜋
𝜃
,
𝐷
)
<
𝜖
}
C 
ϵ
​
 ={θ∈Θ: 
S
^
 (π 
θ
​
 ,D)<ϵ}
Discuss elicitable properties and why CRPS is used as the primary score.

3. Approximate Bayesian Computation for Forecast Evaluation
Review ABC methodology:

Simulation-based rejection sampling targeting 
𝑃
(
𝜃
∣
𝐶
𝜖
,
𝐷
)
P(θ∣C 
ϵ
​
 ,D).

Discuss the asymptotics as 
𝜖
→
0
ϵ→0 and 
𝑞
→
𝜋
q→π.

Discuss identifiability and information loss due to implicit likelihoods.

4. Gibbs-ABC with Block-Wise Parameter Updates
Describe the Gibbs-ABC algorithm (cite and align with Bülte et al., 2025):

Partitioning the parameter vector into blocks 
𝜃
=
(
𝜃
1
,
…
,
𝜃
𝐵
)
θ=(θ 
1
​
 ,…,θ 
B
​
 ).

Sampling conditionally per block given the others.

Define the Markov kernel implicitly induced by the ABC rejection step.

Mathematical description:

Let 
𝜃
=
(
𝛼
𝑏
,
𝛽
𝑏
,
𝛼
𝑠
,
𝛽
𝑠
)
θ=(α 
b
​
 ,β 
b
​
 ,α 
s
​
 ,β 
s
​
 ), for each meteorological variable.

For block 
𝜃
(
𝑏
)
θ 
(b)
 :

𝜃
(
𝑏
)
∼
𝑃
(
𝜃
(
𝑏
)
∣
𝜃
(
−
𝑏
)
,
𝑆
^
(
𝜋
𝜃
,
𝐷
)
<
𝜖
)
θ 
(b)
 ∼P(θ 
(b)
 ∣θ 
(−b)
 , 
S
^
 (π 
θ
​
 ,D)<ϵ)
Emphasize that proposals are accepted only if they decrease (or do not increase) the discrepancy score, approximating the conditional posterior.

Discuss the effect of the number of proposals 
𝐾
K, and theoretical considerations regarding ergodicity and convergence.

5. Implementation and Parallelization
Detail implementation specifics.

Emphasize embarrassingly parallel structure:

Proposals and simulations for each block can be run in parallel.

Parallel CRPS evaluation.

Show compatibility with the cluster setup (submission scripts, PBS, and scratch storage use).

Discuss reproducibility, seed management, and diagnostics.

6. Experimental Setup
Data description (forecast fields, target variables, horizon).

Deterministic forecasting system.

Evaluation metrics.

Posterior summaries, credible intervals.

7. Results
Posterior evolution over Gibbs iterations.

Time-averaged CRPS reduction.

Visuals: posterior distributions of calibration parameters across variables.

Compare different values of 
𝐾
K (proposal count), ensemble size, and Gibbs steps.

8. Theoretical Considerations and Limitations
Discuss:

No guarantee of global improvement in a single step.

Non-exactness due to rejection threshold 
𝜖
>
0
ϵ>0.

Sensitivity to initialisation.

Possible identifiability issues when using only forecast score as surrogate likelihood.

9. Conclusions and Future Work
Formal summary of contributions.

Methodological improvements: adaptive 
𝜖
ϵ, SMC-ABC alternatives.

Extension to continuous-time models or hierarchical priors.

Appendices
Algorithm pseudocode in full detail.

Derivations of scoring rule properties.

Additional plots and diagnostics.