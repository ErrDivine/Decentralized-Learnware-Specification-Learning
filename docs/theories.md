# This is the documentation of the theories of the experiments.

## Assumptions
*Under following assumptions we make the formulation of the approach we address the problem:*

We define the task to be solved as t, variables(the process) as v, A as the set of agents involved and $a_{i}$ as the ith agent.

**Assumption[1]**:As one agent works on the task, it's vote diminishes till some other agent surpass it.  
**Assumption[2]**:The orobability of one agent's taking action is fully dependent on the current task and variables. 

## Core formulations
$$
f_{head}(t,v;\theta,agent) = Pr(action=True|t,v) = vote
$$
$$
f_{agent}(t,v) = \Delta{v}
$$

### Reinforcement Learning for head
#### Basic elements  
**1.$State$**  
All heads share the same $State$
$$
State_t = Vector<t,v>
$$
**2.$Action$**  
Regard the $Action\ space$ as a confidence level ranging from 0 to 1. The policy network output two number $\alpha$ and $\beta$ that constructed the beta distribution for exploration instead of a certain value. Then we sample the $vote_i$ from $Beta(\alpha,\beta)$
$$
f_{head_i}(State_t) = \alpha,\beta 
$$
$$
vote_i = 
\begin{cases}
Beta(\alpha,\beta).sample,\ while\ training \\
\alpha/\alpha+\beta,\ while\ testing
\end{cases}
$$
**3.$Bid\ \&\ Execute$**   
Choose the agent with highest $vote$ to execute
$$
j = argmax_{i} (vote_i)\\
vote_t = vote_j
$$
$\Delta{v}$ here means the executor change
$$
\Delta{v} = f_{agent_j}(t,v)\\
State_{t+1} = State_t + \Delta{v}
$$
Compute $log(pdf(vote_t))$ and prepare for the optimization
$$
\log P_{old} = \text{Beta}(\alpha, \beta).\text{log\_prob}(vote_t)
$$
**4.$Reward$**   
The judger give score if the executor change
$$
score_t = f_{judger}(t,v)
$$
$$
\Delta{score} = score_t - score_{t-1}
$$
Reward $R_t$ for the execute $agent_j$
$$
R_t = \Delta{score} * vote_j
$$
**5.$Buffer$**  
Store for the optimization process
$$
append\ (State_t,vote_t,R_t,log P_{old},j) \ to\ Buffer 
$$

#### Optimization process
**Step 1**  
Pick $(S,vote,R,logP_{old},j)$ from the $Buffer$ and will update parameters of $head_j$, input $S_t$ to $head_j$, get new $(\alpha',\beta')$  
**Step 2**    
Compute the same action $vote$'s $\log P_{new}$ under $\text{Beta}(\alpha', \beta')$  
**Step 3**   
Compute ratio
$$
r(\theta) = \exp(\log P_{new} - \log P_{old})
$$  
**Step 4**    
A critic network predict $\Delta{score}_p$, 
$$
V(S) = \Delta{score}_p * vote_j
$$
$$
A(S) = R - V(S) 
$$
The training process of the critic network can be regarded as a regression network  
**Step 5**  
The PPO lose is
$$
L = -\min \Big( r A, \ \text{clip}(r, 1-\epsilon, 1+\epsilon) A \Big)
$$
Then we do backpropagation and update parameters as usual


## Architecture 
We attempt to concatenate task vector and variables vector for the input of the head module. As they may be 

    t ---encoder--> t`  \
    +                 cross-attention ---nn--> Pr(True|t,v)
    v ---encoder--> v`  /

We consider the heteogenity of task and variables vectors, so head will first build two encoders respectively to process the task and variables vector, so that they can be concatenated later on. Then we obtain cross attention, to get the sequencing information. Then the result of cross attention is put through a feed forward neural network to finally output the probability of the agent, which the head is in charge of, should act(vote).      

And once the agent with the highest vote has changed, we will input the t and v into the agent and let it output till it termininates. Then we check token by token whether the vote's maximum property is violated by other agents, till at some token point the vote of the current agent is not the highest.