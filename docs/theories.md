# This is the documentation of the theories of the experiments.

### Assumptions
*Under following assumptions we make the formulation of the approach we address the problem:*

We define the task to be solved as t, variables(the process) as v, A as the set of agents involved and $a_{i}$ as the ith agent.

**Assumption[1]**:As one agent works on the task, it's vote diminishes till some other agent surpass it.
**Assumption[2]**:The orobability of one agent's taking action is fully dependent on the current task and variables. 

### Core formulations
$$
f_{head}(t,v;\theta,agent) = Pr(action=True|t,v) = vote
$$
$$
f_{agent}(t,v) = \Delta{v}
$$

The reinforement learning part remains to be done.

### Architecture 
We attempt to concatenate task vector and variables vector for the input of the head module. As they may be 

    t ---encoder--> t`  \
    +                 cross-attention ---nn--> Pr(True|t,v)
    v ---encoder--> v`  /

We consider the heteogenity of task and variables vectors, so head will first build two encoders respectively to process the task and variables vector, so that they can be concatenated later on. Then we obtain cross attention, to get the sequencing information. Then the result of cross attention is put through a feed forward neural network to finally output the probability of the agent, which the head is in charge of, should act(vote).      

And once the agent with the highest vote has changed, we will input the t and v into the agent and let it output till it termininates. Then we check token by token whether the vote's maximum property is violated by other agents, till at some token point the vote of the current agent is not the highest.