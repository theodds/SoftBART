# Design document for implementation of Gibbs Priors

We need to make the following changes. First, during Gibbs sampling, we will marginalize out $s$, which will complicate the transition mechanism; it would be too expensive to (say) sample a new $s$ for each tree. Second, we need to make sure that (during the Gibbs sampling step) we appropriately keep track of the transition probabilities.

- During a death move, we will need to reduce the counts by 1
- During a birth move, if we accept, we will need to increase the counts by 1
- For the perturb move, 
- The Prior proposal will require substantial changes, because the prior probability of sampling a tree and its backward transition will be complicated by the decision rules.
- The prior proposal will also need to increment the counts, potentially
  - Maybe this can be built into Sample? Every call to Sample could increment the counts, but then we would need to also manually decrement in a bunch of places
  - Could also pass this in as an option to BirthLeaves whether we want to modify the counts?
- Ultimately, at each stage, we need a function `compute_partition_probability` that we can add into the mix. In general, we could do this in the following way:
  - Write a function called `subtract_tree_counts` that removes from the count vector for a given tree
  - Write a function called `add_tree_counts` 
- Where should all of this logic be encapsulated?

Another possibility would be to keep track of $s$ (for the observed guys) and $D$ during the Gibbs sampler, with a couple of non-zero $s$'s used for $D > Q$. This seems a bit confusing.


# Decisions

- Class specifically for probability updates containing:
  - A vector of counts of the current branch states
  - A map computing V
  - A boolean which indicates whether to use the current state to do the sampling
  
- Should modification to the counts be done when a tree is initialized (as part of the tree growing process), or after a tree is initialized?
  - It should be done AFTER the trees are initialized

- It is the responsibility of the Markov transitions to maintain state. Meaning that:
  - When the forest is initialized, we need to compute the counts
  - Counts must be maintained in the functions DrawPrior, Birth, Death, and Perturb
  
- Changes to initialization of the algorithm
  - All counts set equal to 1 so that everything has equal probability? Equivalently, 
  - 


