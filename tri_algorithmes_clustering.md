# Clustering algorithms

## Common Notation

To keep the explanations concise, the following variables are used consistently across the algorithms described below:

* **$n$:** The total number of data points (samples).
* **$d$:** The dimensionality of the feature space (number of features per sample).
* **$k$:** The number of clusters (when specified or predetermined).
* **$t$:** The number of iterations required until the algorithm converges.
* **$X$:** The input data matrix of shape ($n \times d$).
* **$x$:** A single data point or sample vector.
* **$\mu$:** The centroid, exemplar, or candidate position representing a cluster.
* **$S$:** The set of clusters. 

## K-Means (Lloyd)

* **Parameters:** number of clusters
* **Scalability:** Very large n_samples, medium n_clusters with MiniBatch code
* **Usecase:** General-purpose, even cluster size, flat geometry, not too many clusters, inductive
* **Geometry:** Distances between points


### What? How?
#### What is the algorithm trying to find or optimize?
Partition $n$ samples into $k$ distinct sets in a way that minimizes the **Within-Cluster Sum of Squares (WCSS)** (basically variance).
Formally, the objective function the algorithm seeks to find is:

$$ \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_{i}} ||x - \mu_{i}||^{2} $$

$|| \cdot ||$ is $L^2$ norm.

#### Algorithm overview
1. Choose the initial centroids (many methods, random, arbitrary by user, or K-means++,...)
2. Loop:
    1. Assign each sample to its nearest centroid
        * If a cluster receives no samples, the algorithm must define an empty-cluster policy before recomputing centroids, such as keeping the previous centroid, reinitializing it, or relocating it to a high-error sample.
    2. Create new centroids by taking the mean value of all of the samples assigned to each previous centroid

    Stop the loop when the distance between a previous centroid and the new one is less than a threshold.

#### Time complexity?
Running time of $O(nkdt)$.

Worst case lower bound of $2^{(\Omega(\sqrt{n}))}$ iterations. Upper bound of $O(k^n)$ iterations because no partition of points into clusters is ever repeated. Despite this, the alg is highly popular due to its observed speed in practical applications, where the required number of iterations is generally much less than the total number of points.

#### Space complexity?
$O((n+k)d)$

#### What are the heaviest mathematical operations?
* Compute $k$ distances for $n$ vectors at step 2.1.
* Compute $k$ means of in total $n$ vectors at step 2.2.
* Compute $k$ distances to check if we end the loop (way less heavy)

#### Data dependency on looping operation?
Dependant between iterations (duh), but independant within an iteration.

#### Nested control flow/branching?
No

#### Irregular Whiles ?
No

#### Memory access pattern? (contiguous, strided, random)
Should be entirely contiguous for step 2.1.
Step 2.2 wouldn't.

#### Compute-bound or memory-bound?
Very memory-bound if $k$ (and $d$) are small compared to $n$, but could be compute-bound else, when computing all the distances at step 2.1.

#### Reductions necessary?
* Sum-reduction during the mean computation for the new centroids (step 2.2.).
* Sum-reduction over dimensions when computing squared Euclidean distances or dot products.
* Min/argmin-reduction over the k centroids to choose the nearest centroid for each sample.


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
Yup: <https://fukushima.web.nitech.ac.jp/paper/2021_iwait_otsuka.pdf> for AVX/AVX2

#### Are there example implementations?
Yes: <https://github.com/Deniskore/kmeans_uni> in Rust

#### Or maybe there are GPU versions?
CPU-GPU version: <https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/cpu_gpu_kmeans.html>


### Sources

* <https://en.wikipedia.org/wiki/K-means_clustering>
* https://scikit-learn.org/stable/modules/clustering.html#k-means

### Conclusion: Should we vectorize it in EVE?

Yes, it's an easy algorithm by itself, there's no major blocking point, there are already SIMD versions of it, and it's very popular (although has limited use).

### Variants of the algorithm
* **Minibatch K-Means**: reduces computation time by using subsets of the input data randomly sampled in each iteration. Produces very slightly worse results but there shouldn't be any problem for vectorization compared to original. And there's already an implementation in the `kmeans_uni` repo. So we should probably add it.

* **Elkan**: Can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it's more memory intensive due to the allocation of an extra array of shape `(n_samples, n_clusters)`.
It will be probably harder to vectorize because more control flow, and the fact that it's more memory-intensive is counter-productive.
So I'd say it should be low priority.

* **Bisecting K-Means**: https://www.philippe-fournier-viger.com/spmf/bisectingkmeans.pdf iterative variant using divisive hierarchical clustering. More efficient when the number of clusters is large. Should be fine to vectorize.


## Affinity propagation

* **Parameters:** damping, sample preference
* **Scalability:** Not scalable with n_samples
* **Usecase:** Many clusters, uneven cluster size, non-flat geometry, inductive
* **Geometry:** Graph distance (e.g. nearest-neighbor graph)

### What? How?
#### What is the algorithm trying to find or optimize?
Instead of requiring the number of clusters to be specified beforehand, Affinity Propagation seeks to find a small number of **exemplars** (samples that are most representative of other samples) and creates clusters around them. 

It optimizes this by iteratively sending and updating messages between pairs of data points. These messages represent the accumulated evidence for the suitability of one sample to be the exemplar of another, factoring in a *preference* parameter (which influences how many exemplars are chosen) and the similarities between the points.

#### Algorithm overview
1. Initialize all responsibility $R_{x, \mu}$ and availability $A_{x,\mu}$ values to zero.
2. Loop:
    1. **Update Responsibilities:** Calculate the accumulated evidence that sample $\mu$ should be the exemplar for sample $x$. 
       $$R_{x,\mu} \leftarrow S_{x,\mu} - \max_{\mu' \neq \mu} [ A_{x,\mu'} + S_{x,\mu'} ]$$
       Where $S_{x,\mu}$ is the similarity (e.g., negative squared euclidean distance) between samples $x$ and $\mu$.
    2. **Update Availabilities:** Calculate the accumulated evidence that sample $x$ should choose sample $\mu$ to be its exemplar.
       $$A_{x,\mu} \leftarrow \min \left[ 0, R_{\mu, \mu} + \sum_{x' \notin \{x, \mu\}} \max(0,R_{x', \mu}) \right]$$
        OR for diagonal elements:
       $$A_{\mu, \mu} \leftarrow \sum_{x' \neq \mu} \max(0, R_{x', \mu})$$
    
    3. **Apply Damping:** To avoid numerical oscillations during the updates, apply a damping factor $\lambda$ to the new messages, weighting them against the messages from the previous iteration $t$:
       $${R_{t+1}}_{x,\mu} = \lambda \cdot {R_{t}}_{x,\mu} + (1-\lambda) \cdot {R_{new}}_{x,\mu}$$
       $${A_{t+1}}_{x,\mu} = \lambda \cdot {A_{t}}_{x,\mu} + (1-\lambda) \cdot {A_{new}}_{x,\mu}$$

    Stop the loop when the algorithm converges (no change in the number of estimated clusters for a set number of iterations) or when the maximum number of iterations is reached.

3. Choose the final exemplars based on the converged responsibility and availability messages to define the final clustering.


#### Time complexity?
$\mathcal{O}(n^2 t)$

#### Space complexity?
$\mathcal{O}(n^2)$ if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used.
But we're (obviously?) going to use a dense matrix for vectorization.

#### What are the heaviest mathematical operations?
* Compute $n^2$ similarities (e.g., negative squared Euclidean distances) between all pairs of the $n$ samples before the loop begins.
* Compute $n^2$ responsibility updates at step 2.1. This heavily relies on finding the maximum value across $n$ elements for every pair (though often optimized in practice by finding only the largest and second-largest values per row).
* Compute $n^2$ availability updates at step 2.2. This involves applying a $\max(0, \cdot)$ filter and summing across up to $n$ elements for each column.
* Perform element-wise damping on the two $n \times n$ matrices at step 2.3, which requires multiple scalar multiplications and additions across $2n^2$ elements per iteration.

#### Data dependency on looping operation?
Between iterations, and in an operation updating matrix $A$ requires having updated matrix $R$.

#### Nested control flow/branching?
Probably not.

#### Irregular Whiles ?

No

#### Reductions necessary?
* Sum-reduction in init. step
* Max-reduction in step 2.1.
* Sum-reduction in step 2.2.
* Bool-reduction in convergence check

#### Random Memory access patterns (or Strided)?
In step 2.2, computing $A_{x,μ}$​ requires a sum over $x′$ for $R_{x′,μ}$. So we are reading down the columns of $R$. In a row-major layout, we access memory through a stride of n.

#### Compute-bound or memory-bound?
Memory bound


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
Didn't find any

#### Are there example implementations?
https://github.com/cjneely10/affinityprop
This readme also says this:
> An existing sklearn version is implemented using the Python library numpy which incorporates vectorized row operations. Coupled with SIMD instructions, this results in decreased time to finish.


#### Or maybe there are GPU versions?
https://pmc.ncbi.nlm.nih.gov/articles/PMC5650075/pdf/nihms872315.pdf
https://github.com/jnuez94/Affinity-Propagation-GPU


### Sources

* https://en.wikipedia.org/wiki/Affinity_propagation
* https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
* https://dashee87.github.io/data%20science/general/Clustering-with-Scikit-with-GIFs/


### Conclusion: Should we vectorize it in EVE?

Maybe, it could be worse.


### Variants of the algorithm

https://www.ijcai.org/Proceedings/11/Papers/373.pdf
Probably much harder to vectorize: segmented reductions for each node: they have variable number of neighbors. We have a sparse graph instead of the matrices,... But it's faster ($O(n^2 + Et)$ with $E \ll n^2$) and guarantees the exact same result.

## Mean-shift

* **Parameters:** bandwidth
* **Scalability:** Not scalable with n_samples
* **Usecase:** Many clusters, uneven cluster size, non-flat geometry, inductive
* **Geometry:** Distances between points


### What? How?
#### What is the algorithm trying to find or optimize?
Mean Shift is a centroid-based, mode-seeking algorithm that aims to discover blobs in a smooth density of samples by locating the maxima of a probability density function. 

It optimizes the position of candidate centroids using a "hill climbing" technique, finding local maxima of the estimated probability density by iteratively shifting candidates to the mean of the points within their neighborhood. The centroid is shifted by a mean shift vector that always points towards the direction of the maximum increase in the density of points.

#### Algorithm overview
1. Initialize the centroid candidates $\mu$ (typically starting with all data points $x$ simultaneously).
2. Loop:
    1. **Compute Mean:** Calculate the weighted mean $m(\mu)$ of the density in the window (neighborhood $N(\mu)$) determined by a kernel function $K$ (such as a Flat or Gaussian kernel).
    2. **Shift:** Update the centroid candidate's position to the newly calculated mean.

    Stop the loop when $m(\mu)$ converges.

3. **Post-Processing:** Filter the candidates to eliminate near-duplicates, merging the points that converged on the same peak to form the final set of centroids.

#### Time complexity?
In the standard, naive implementation, the total time complexity is $\mathcal{O}(t n^2 d)$. This quadratic scaling occurs because every single point must evaluate its distance to every other point to compute the kernel-weighted mean during every iteration.

However, various "Fast Mean Shift" strategies are heavily utilized in practice to reduce this bottleneck:
* **Tree-Based Pruned Search:** By structuring the data with k-d trees or ball trees to limit neighbor searches to a local radius, the complexity drops to $\mathcal{O}(t n \log n d)$ in low to moderate dimensions. This can still degrade toward the worst-case if applied to very high dimensions.
* **Approximate Nearest Neighbors (ANN):** Using techniques like Locality-Sensitive Hashing (LSH) avoids exhaustive distance calculations, reducing the complexity to near-linear time (often $\mathcal{O}(t n \log n)$ or even $\mathcal{O}(t n)$), which scales much better for high-dimensional data at the cost of slight approximation.
* **Fast Gauss Transform (FGT):** This method approximates the influence of distant points using Hermite expansions, bringing the complexity down to approximately $\mathcal{O}(tn)$ for low-dimensional data. 
* **Seed Point Sampling:** By only running the shifting trajectories on a representative subset of seeds ($s$) instead of the whole dataset, the dominant mode-seeking cost is reduced to $\mathcal{O}(s n d t)$.

#### Space complexity?
$\mathcal{O}(nd)$

However, if we implement "Fast Mean Shift" strategies to speed up the time complexity, they generally introduce additional memory overheads:
* **Tree-Based Pruned Search:** Building spatial indexing structures like k-d trees or ball trees over the reference set requires an additional $\mathcal{O}(n)$ memory.
* **Approximate Nearest Neighbors (LSH):** Requires additional memory to construct and store the $L$ independent hash tables used to group the candidate neighbors.
* **Fast Gauss Transform (FGT):** Requires extra storage to hold the vector of Hermite expansion coefficients that are precomputed for each grid cell.

#### What are the heaviest mathematical operations?
* Compute $n^2$ distances between all $n$ centroid candidates and all $n$ data points at step 2.1.
* Compute $n^2$ kernel evaluations to determine the weight of each point's contribution at step 2.1.
* Compute $n$ weighted means at step 2.1, which involves multiplying the $n$ data point vectors by their scalar kernel weights, summing the results, and dividing by the sum of the weights for each candidate.
* Compute $n$ distances between the previous and updated centroid candidate positions to check against the convergence threshold to determine if we end the loop.

#### Data dependency on looping operation?
Dependant between iterations, but the update operations for each data point are completely independent within a single iteration.

#### Nested control flow/branching?
No

#### Irregular Whiles ?
I think yes: the iteration loop will stop much faster for some centroid candidates than some other. So we'd need to repack the non-yet-idle points into a new vector to avoid running a vector with almost all idle (so masked) points just to iterate on a single point in it.

#### Reductions necessary?
* Sum-reduction in step 2.1 (calculating the sums for the numerator and denominator of the kernel-weighted mean)
* Bool-reduction in the convergence check
* Min-reduction in the final cluster assignment step

#### Random Memory access patterns (or Strided)?
No

#### Compute-bound or memory-bound?
Compute-bound (distances, kernel evaluations)


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
No, but there's a paper that is stating that:
> Due to its histogramic nature, mean shift is not normally considered worth optimizing with vector instructions.

Although it's not on clustering specifically.

#### Are there example implementations?

No

#### Or maybe there are GPU versions?

https://github.com/masqm/Faster-Mean-Shift-Euc

### Sources

* https://scikit-learn.org/stable/modules/clustering.html#mean-shift
* https://en.wikipedia.org/wiki/Mean_shift
* https://www.mdpi.com/2227-7390/13/21/3408
* https://ieeexplore.ieee.org/document/1488889

### Conclusion: Should we vectorize it in EVE?

The irregular while might be dealbreaker.
The lack of example implementations makes it harder than others.
So lower priority.

### Variants of the algorithm

Described in the complexity sections.


## Spectral clustering

* **Parameters:** number of clusters
* **Scalability:** Medium n_samples, small n_clusters
* **Usecase:** Few clusters, even cluster size, non-flat geometry, transductive
* **Geometry:** Graph distance (e.g. nearest-neighbor graph)


### What? How?
#### What is the algorithm trying to find or optimize?
> SpectralClustering performs a low-dimension embedding of the affinity matrix between samples, followed by clustering, e.g., by KMeans, of the components of the eigenvectors in the low dimensional space. It is especially computationally efficient if the affinity matrix is sparse and the amg solver is used for the eigenvalue problem (Note, the amg solver requires that the pyamg module is installed.)

needs to compute eigenvectors ==> I'm skipping it for now.

#### Algorithm overview



#### Time complexity?



#### Space complexity?



#### What are the heaviest mathematical operations?



#### Data dependency on looping operation?



#### Nested control flow/branching?



#### Irregular Whiles ?



#### Reductions necessary?



#### Random Memory access patterns (or Strided)?



#### Compute-bound or memory-bound?




### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?



#### Are there example implementations?



#### Or maybe there are GPU versions?
https://theses.hal.science/tel-04114475/

### Sources

* https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

### Conclusion: Should we vectorize it in EVE?



### Variants of the algorithm




## Ward hierarchical / Agglomerative clustering

* **Parameters:** number of clusters or distance threshold, linkage type, distance
* **Scalability:** Large n_samples and n_clusters
* **Usecase:** Many clusters, possibly connectivity constraints, non Euclidean distances, transductive
* **Geometry:** Any pairwise distance


### What? How?
#### What is the algorithm trying to find or optimize?
> Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample. See the Wikipedia page for more details. \
The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy: \
    - Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach. \
    - Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters. \
    - Average linkage minimizes the average of the distances between all observations of pairs of clusters. \
    - Single linkage minimizes the distance between the closest observations of pairs of clusters.

So basically Ward is just a specific type of linkage of Agglomerative.

#### Algorithm overview
Agglomerative clustering is a "bottom-up" hierarchical approach.

1. **Initialize:** Begin with each individual data point forming its own separate cluster. Optionally, compute an initial $n \times n$ distance matrix where each cell represents the pairwise distance between the $i$-th and $j$-th elements.
2. **Loop:**
    1. **Find Nearest:** Identify the two most similar (closest) clusters based on the chosen distance metric and a specific linkage criterion.
    2. **Merge:** Combine these two closest clusters into a single new cluster. These merges are determined in a greedy manner.
    3. **Update:** Update the distance matrix by merging the corresponding rows and columns and calculating the new distances between the newly formed cluster and all other remaining clusters.

    Stop the loop when all data points are combined into a single overarching cluster, or when a pre-determined stopping criterion (such as a target number of clusters or a specific distance threshold) is met.

3. **Post-Processing:** The resulting hierarchy is typically presented as a dendrogram. To obtain a specific partitioning of the data, "cut" the dendrogram tree at a selected height or at a large vertical gap.


#### Time complexity?
$\mathcal{O}(n^3)$. This cubic scaling occurs because at each of the n-1 merge steps, the algorithm must search and update an $n \times n$ proximity matrix to find the closest clusters. 

However, various optimized strategies are heavily utilized in practice to reduce this bottleneck:
* **Nearest-Neighbor Chains and Priority Queues:** For many linkage criteria, utilizing these data structures allows the algorithm to achieve $\mathcal{O}(n^2 log n)$ time complexity in practice.
* **Heap-Based Search:** Using a heap reduces the general case runtime to $\mathcal{O}(n^2 log n)$, though this comes at the cost of additional memory requirements.
* **Optimal Special Cases (SLINK/CLINK):** For specific linkage types, optimal efficient methods exist. The SLINK algorithm for single-linkage and the CLINK algorithm for complete-linkage reduce the time complexity down to exactly $\mathcal{O}(n^2)$.
* **Quadtrees:** Certain methods utilize quadtrees to demonstrate an $\mathcal{O}(n^2)$ total running time.
* **Scalable Density Approaches (e.g., HAL-x):** For massive, high-dimensional datasets where $\mathcal{O}(n^2)$ is still too slow, algorithms like HAL-x use downsampling, initial density clustering in low-dimensional space, and trained classifiers to determine merges, bypassing the need for exact geometric linkage updates on the full dataset.

#### Space complexity?
$\mathcal{O}(n^2)$. The algorithm requires this memory to store the full matrix of pairwise distances between all observations. This quadratic memory requirement makes the standard algorithm too slow and difficult to apply to even medium-sized datasets.

Various optimizations and alternative implementations handle memory differently:
* **Quadtrees:** Methods existing that use quadtrees can reduce the memory requirement down to $\mathcal{O}(n)$ space.
* **Heap Overheads:** While using a heap improves time complexity, the memory overheads can be too large to make it practically usable in many scenarios.
* **Subsampling Workarounds:** Scalable implementations like HAL-x deliberately restrict the expensive matrix calculations to manageable downscaled subsamples, which is extremely useful when the hierarchical clustering process requires minimal RAM.

#### What are the heaviest mathematical operations?
* Compute $n^2$ pairwise distances between all $n$ data points at step 1.
* Compute $n-1$ global minimums to search the proximity matrix at step 2.1.
* Compute up to $n$ distance updates per iteration at step 2.3. Depending on the chosen linkage method, this requires evaluating recursive Lance-Williams equations (which consist of scalar multiplications and additions) or recomputing slower full formulas to find the new distance between the newly merged cluster and all other remaining clusters.

#### Data dependency on looping operation?
Yes, heavily.
> At each step, the algorithm merges the two most similar clusters... This process continues until all data points are combined into a single cluster

#### Nested control flow/branching?
No

#### Irregular Whiles ?
No

#### Reductions necessary?
Finding a global minimum at each step.

#### Random Memory access patterns (or Strided)?
Yes
> rows and columns are merged as the clusters are merged and the distances updated

#### Compute-bound or memory-bound?
Memory-bound. \
The algorithm must repeatedly search the $\mathcal{O}(n^2)$ proximity matrix to find the global minimum and then update specific rows and columns associated with the newly merged cluster. This leads to continuous, scattered reads and writes across a large memory footprint. As clusters merge, the active elements in the matrix become sparse or require significant memory shifting to remain contiguous, leading to poor cache utilization and making memory access latency the primary bottleneck, rather than raw mathematical computation.


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
Nah

#### Are there example implementations?
No

#### Or maybe there are GPU versions?
* https://arxiv.org/html/2306.16354v1
* https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=1608&context=etd This one's a distributed alg.


### Sources

* https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
* https://en.wikipedia.org/wiki/Hierarchical_clustering

### Conclusion: Should we vectorize it in EVE?
No

### Variants of the algorithm




## DBSCAN

* **Parameters:** neighborhood size, minimum nbr of pts
* **Scalability:** Very large n_samples, medium n_clusters
* **Usecase:** Non-flat geometry, uneven cluster sizes, outlier removal, transductive
* **Geometry:** Distances between nearest points


### What? How?
#### What is the algorithm trying to find or optimize?
DBSCAN is a non-parametric, density-based clustering algorithm that groups closely packed points together while marking points in low-density regions as outliers. It optimizes a loss function to minimize the total number of clusters, subject to the condition that every pair of points within a single cluster must be density-reachable. This optimization formally enforces the algorithm's core cluster properties of "maximality" and "connectivity." 

#### Algorithm overview
DBSCAN operates using a sequential, query-based approach to build clusters. 

1. **Initialize:** Define the `eps` (neighborhood radius) and `minPts` (minimum number of points) parameters. Start the algorithm with an arbitrary data point that has not yet been visited. 
2. **Loop:**
    1. **Find Neighbors:** Retrieve the `eps`-neighborhood of the current point. 
    2. **Density Check:** If the neighborhood contains at least `minPts` points, the point is considered a core point and a new cluster is started. If the neighborhood does not contain sufficiently many points, the point is labeled as noise. This noise label can change if the point is later found within the `eps`-neighborhood of a different core point. 
    3. **Expand Cluster:** Add all points found within the `eps`-neighborhood to the new cluster. If any of these newly added points are also dense core points, add their own `eps`-neighborhoods to the cluster. 

    Stop the loop when the current density-connected cluster is completely found. Then, retrieve a new unvisited point to process until all points in the database have been visited. 



#### Time complexity?
Worst case is $\mathcal{O}(n^2)$. Depending on the distance function, it can be $\mathcal{O}(n^2d)$ or even $\mathcal{O}(n^3)$.
Can drop to $\mathcal{O}(n\log{n})$ with KD-tree.

#### Space complexity?
$\mathcal{O}(n)$ but some implementation can go up to $\mathcal{O}(n^2)$.

#### What are the heaviest mathematical operations?
* Compute up to $n$ distance evaluations for each point (totaling up to $n^2$ pairwise distance computations if using a linear scan without an index) to identify the $\epsilon$-neighborhood at step 2.1.
* Compute the neighborhood size $|N|$ up to $n$ times to evaluate if it meets the $minPts$ density threshold at step 2.2.
* Compute exactly $n$ total neighborhood range queries across the entire algorithm execution to find the density-connected components at step 2.3.

#### Data dependency on looping operation?
In the outer loop (we choose an unprocessed point each time), but also in the cluster expansion process (we compute the ranges, find the points, add them, then recompute the ranges)

#### Nested control flow/branching?
I don't think so.

#### Irregular Whiles ?
I mean that really depends on what we'll be doing. As both loops are nested and sequential in the original alg, if we for example expand multiple clusters at the same time, some might finish earlier than others.
A lot of conditional loopings also.

#### Reductions necessary?
* Sum-reduction during the distance computation in the `RangeQuery` step.
* Sum-reduction in the density check step.
* Bool-reduction in the inner loop continuation check.

#### Random Memory access patterns (or Strided)?
* **Random accesses (Gathers) from the SeedSet:** Points added to the expansion queue are based on spatial proximity, not memory layout. When we pop a point from the `SeedSet` to process it, its original index in the dataset ($X$) is effectively random. 
* **Random updates (Scatters) to the Label array:** When we mark a neighborhood of points as belonging to a cluster or as visited, we are scattering state updates to arbitrary indices in $L$.
* **Tree Traversal / Pointer Chasing (if indexed):** If we try to optimize the `RangeQuery` using a spatial index (like a kd-tree or R-tree) rather than a naive linear scan, the memory access becomes heavily random and involves unpredictable pointer chasing. If we stick to the naive linear scan to keep memory access sequential, we keep SIMD happy but pay the $O(n^2)$ compute penalty.

#### Compute-bound or memory-bound?
Memory-bound (only if we pre-compute the distances?)


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
VHPDBSCAN: https://www.techrxiv.org/doi/abs/10.36227/techrxiv.171085046.60925150/v2

#### Are there example implementations?
https://github.com/weilun-chiu/dbscan

#### Or maybe there are GPU versions?



### Sources

* https://en.wikipedia.org/wiki/DBSCAN#Complexity
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
* https://dl.acm.org/doi/10.1145/3068335

### Conclusion: Should we vectorize it in EVE?

Yes! Would be more challenging than KMeans but has already been done.

### Variants of the algorithm

* The index-based one.
* **sDBSCAN**: https://arxiv.org/pdf/2402.15679. uses random projections to estimate the $\varepsilon$-neighborhood without a KD-tree.
We should probably implement its aspects into our algorithm.
* **HPDBSCAN**: the original parallel version https://www.researchgate.net/publication/301463871_HPDBSCAN_highly_parallel_DBSCAN
* **VHPDBSCAN**
* **G-DBSCAN**: https://www.sciencedirect.com/science/article/pii/S1877050913003438 early GPU-accelerated version
* **BPS-HDBSCAN**: https://jan.ucc.nau.edu/mg2745/publications/Gowanlock_ICS2019.pdf CPU/GPU co-op
* **Wang et al. (2020) Parallel DBSCAN**: https://arxiv.org/pdf/1912.06255 Work-efficient parallel algorithm


## HDBSCAN

* **Parameters:** minimum cluster membership, minimum point neighbors
* **Scalability:** large n_samples, medium n_clusters
* **Usecase:** Non-flat geometry, uneven cluster sizes, outlier removal, transductive, hierarchical, variable cluster density
* **Geometry:** Distances between nearest points


### What? How?
#### What is the algorithm trying to find or optimize?
HDBSCAN is a density-based, hierarchical clustering method that generates a complete clustering hierarchy from which a simplified tree of significant clusters can be extracted. Unlike DBSCAN, it is capable of finding clusters of very different densities without relying on a single global density threshold. Specifically, the algorithm optimizes the extraction of a "flat" partition by formalizing the task as an optimization problem where the objective is to maximize the overall stability of the selected clusters. This cluster stability is based on a relative excess of mass, measuring how long a cluster "survives" as the density threshold changes before splitting or disappearing.

#### Algorithm overview
HDBSCAN conceptually builds upon DBSCAN* (a variant where border points are considered noise) by computing a hierarchy of all possible DBSCAN* partitions. The algorithm operates in the following sequential steps:

1. **Calculate Core Distances:** Compute the core distance for all data objects in the dataset with respect to the parameter `m_pts`, which is the distance to an object's `m_pts`-nearest neighbor.
2. **Construct Mutual Reachability Graph:** Conceptually, build a complete graph where the weight of each edge is the mutual reachability distance between two points, defined as the maximum of their respective core distances and the actual distance between them.
3. **Build Minimum Spanning Tree (MST):** Compute an MST of the Mutual Reachability Graph.
4. **Extend MST:** Add a "self edge" to each vertex in the MST, using the core distance of the corresponding object as the edge weight.
5. **Extract Hierarchy:** Iteratively remove all edges from the extended MST in decreasing order of their weights to build a dendrogram. Components that become disconnected represent splits in the cluster hierarchy, with new clusters formed if they contain at least one edge, else they are labeled as noise.
6. **Condense the Tree:** Simplify the hierarchy by setting a minimum cluster size, denoted as `m_clSize`. When a cluster splits, any resulting component with fewer than `m_clSize` objects is considered a spurious component and labeled as noise, meaning the main cluster simply shrunk rather than undergoing a "true" split.
7. **Extract Optimal Clustering:** To find the optimal flat partition, calculate the stability of each cluster in the condensed tree based on its relative excess of mass. Then, traverse the tree bottom-up to select a non-overlapping set of clusters that maximizes the total sum of stabilities.

#### Time complexity?
$\mathcal{O}(dn^2)$, or $\mathcal{O}(n^2)$ if we precompute the distances.

#### Space complexity?
$\mathcal{O}(dn)$, or $\mathcal{O}(n^2)$ if we precompute the distances.

#### What are the heaviest mathematical operations?
* Compute $n$ total $m_{pts}$-nearest neighbor queries (or a full $n \times n$ distance matrix) to establish the core distances for every data object.
* Compute the Minimum Spanning Tree (MST) of the Mutual Reachability Graph; using Prim’s algorithm with a linear scan, this involves $n$ iterations of finding the minimum weight edge, where each edge weight is a triple-max evaluation: $\max(d_{core}(p), d_{core}(q), d(p, q))$.
* Sort the $n-1$ edges of the MST in decreasing order of weights to iteratively construct the dendrogram.
* Evaluate the Relative Excess of Mass (stability) for every node in the condensed tree by integrating the density survival (difference between the level a point falls out of a cluster and the level the cluster was born) across all members.
* Perform a bottom-up recursive optimization to maximize the total sum of stabilities across the hierarchy.

#### Data dependency on looping operation?



#### Nested control flow/branching?



#### Irregular Whiles ?



#### Reductions necessary?



#### Random Memory access patterns (or Strided)?



#### Compute-bound or memory-bound?




### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
https://github.com/MartinTschechne/ASL-hdbscan/blob/master/33_report.pdf

#### Are there example implementations?
Yes!! https://github.com/MartinTschechne/ASL-hdbscan

#### Or maybe there are GPU versions?



### Sources

* https://scikit-learn.org/stable/modules/clustering.html#id11
* https://en.wikipedia.org/wiki/DBSCAN
* https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14


### Conclusion: Should we vectorize it in EVE?

Yes! We already have a super nice implementation

### Variants of the algorithm

* **Accelerated HDBSCAN***: https://arxiv.org/pdf/1705.07321 Better time complexity, but inherently serial (in their own words). They are saying it could be possible to parallelize it (they have leads), but that would be creating a new algorithm, which is very probably out of scope.
* **PHDBSCAN**: https://ualberta.scholaris.ca/server/api/core/bitstreams/753edc02-dca8-4695-af47-a2f6ac58270b/content Parallel version, optimizes parallelism with map-reduces.


## OPTICS

* **Parameters:** minimum cluster membership
* **Scalability:** Very large n_samples, large n_clusters
* **Usecase:** Non-flat geometry, uneven cluster sizes, variable cluster density, outlier removal, transductive
* **Geometry:** Distances between points


### What? How?
#### What is the algorithm trying to find or optimize?

OPTICS (Ordering points to identify the clustering structure) is a density-based algorithm that generalizes DBSCAN to address the problem of detecting meaningful clusters in data of varying density. Instead of generating a single flat clustering assignment, it linearly orders the database points such that spatially closest points become neighbors in the ordering. By assigning each point a "core distance" and a "reachability distance," OPTICS produces a hierarchical representation of the data. This hierarchy is often visualized as a reachability plot (a special kind of dendrogram) where point density is represented on the Y-axis and dense clusters appear as deep "valleys".

#### Algorithm overview

OPTICS shares DBSCAN's core logic but maintains known, unprocessed cluster members in a priority queue (indexed heap) rather than a simple set.

1. **Initialize:** Define `eps` (or `max_eps` to limit runtime and serve as a maximum search radius) and `MinPts`. Start with an arbitrary unprocessed point and retrieve its `eps`-neighborhood.
2. **Calculate Core Distance:** If the neighborhood contains at least `MinPts` points, the point is considered a core point. Its "core distance" is the exact distance to its `MinPts`-th closest neighbor.
3. **Expand & Update Priority Queue:** Add the core point's neighbors to a priority queue. For each neighbor, calculate its "reachability distance," which is the maximum of the core point's core distance or the actual distance between the two points.
4. **Heap Maintenance:** If a neighbor is already in the priority queue but the newly calculated reachability distance is smaller, update its value to the new distance and move it up the heap.
5. **Output Ordering:** Mark the current point as processed and append it to the ordered output list. Pop the next point with the lowest reachability distance from the priority queue and repeat the expansion process.
6. **Cluster Extraction:** Once the reachability graph is built, clusters can be extracted either by cutting the reachability plot at a specific threshold to mimic DBSCAN, or by identifying steep valleys using the `xi` parameter.

#### Time complexity?

* The worst-case time complexity is $\mathcal{O}(n^2)$, exactly like DBSCAN.
* If an efficient spatial index is used, the overall runtime can be reduced to $\mathcal{O}(n \log n)$ because neighborhood queries can be executed more quickly. However, this improvement heavily relies on `max_eps` being chosen appropriately. If it is set to infinity or larger than the maximum distance in the dataset, the complexity degrades back to $\mathcal{O}(n^2)$ since every neighborhood query will return the full dataset.
* While a single run of OPTICS has a constant slowdown factor (reported as 1.6 by the original authors) compared to DBSCAN, OPTICS requires less cumulative runtime if you need to evaluate clustering structures across many different $\epsilon$ values. 

#### Space complexity?

* The space complexity maintains $\mathcal{O}(n)$ scaling.
* Unlike implementations that require materializing a full distance matrix (which would cost $\mathcal{O}(n^2)$ memory), OPTICS is generally designed to use spatial indexing trees to remain memory-efficient on large datasets.
* It requires $\mathcal{O}(n)$ memory to store the output ordering, the reachability-distances, and the core-distances, as well as to maintain the priority queue (indexed heap) of known but unprocessed cluster members.

#### What are the heaviest mathematical operations?

* Compute $n$ total $\epsilon$-neighborhood range queries to find neighbors across the entire algorithm execution.
* Compute the distance to the $MinPts$-th nearest neighbor within each neighborhood to establish the core-distances.
* Evaluate the $\max$ function between a core point's core-distance and its actual spatial distance to a neighbor to continuously update reachability-distances.
* Dynamically evaluate and sort the minimum reachability distances of all unprocessed neighbors residing in the priority queue.


#### Data dependency on looping operation?

* **Priority Queue State:** The algorithm is inherently sequential. The choice of the next point to process in the inner cluster-expansion loop is strictly dependent on which point currently holds the minimum reachability distance in the `Seeds` priority queue. You cannot determine the next point to process until the current point's neighborhood has been fully evaluated and the queue is updated.
* **Reachability Updates:** There is a direct read-compare-write dependency when evaluating neighbors. A point's reachability distance is overwritten, and its position in the queue moved up, only if the newly calculated reachability distance is strictly smaller than its currently stored reachability distance.

#### Nested control flow/branching?

Yes:
* The main loop contains an `if` check for valid core-distances, which itself conditionally triggers an inner loop over the `Seeds` priority queue.
* Within the priority queue expansion, there is another `if` check to see if the newly popped point is a core point.
* The heaviest nesting occurs during the queue `update` step: it iterates through neighbors, checks `if` the neighbor is unprocessed, and if true, hits a nested `if/else` block to check if the reachability distance is `UNDEFINED`. Inside the `else` branch, there is yet another nested `if` to check if the newly calculated reachability distance improves upon the currently stored one.

#### Irregular Whiles?

Yes. The cluster expansion loop (`for each next q in Seeds do`) acts as a highly irregular `while (!Seeds.isEmpty())` loop. Its duration is completely data-dependent; it will run for thousands of iterations when expanding a massive, dense cluster, but won't execute at all if the initial queried point evaluates to `UNDEFINED` for its core-distance. 

#### Reductions necessary?

* **Sum-reduction:** Necessary during the underlying distance calculations to evaluate the `eps`-neighborhood, identical to DBSCAN.
* **Sort/Selection-reduction:** Required to find the `MinPts`-th smallest distance within a neighborhood to compute a point's `core-distance`.
* **Max-reduction:** Used when computing the new reachability distance, requiring an evaluation of $\max(\text{core-dist}, \text{dist}(p,o))$.
* **Min-reduction:** Required to continuously extract the point with the minimum reachability distance from the priority queue (the heap's `extract-min` operation).

#### Random Memory access patterns (or Strided)?

* **Random accesses (Gathers) driven by the Priority Queue:** The inner loop's processing order is dictated by extracting the point with the minimum reachability distance from the priority queue (e.g., an indexed heap). Consequently, fetching the next point's data (like its coordinates or core distance) results in highly random memory jumps, completely decoupled from the original dataset's layout.
* **Random updates (Scatters) to Reachability States:** When expanding a core point's neighborhood, the algorithm updates the `reachability-distance` of specific neighbors. It also modifies their positions in the priority queue via a `move-up` operation. This scatters state updates to arbitrary indices across memory based purely on spatial proximity.

#### Compute-bound or memory-bound?

Memory-bound, probably more than DBSCAN.

### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?

No

#### Are there example implementations?

No

#### Or maybe there are GPU versions?

Not directly

### Sources

* https://scikit-learn.org/stable/modules/clustering.html#optics
* https://en.wikipedia.org/wiki/OPTICS_algorithm

### Conclusion: Should we vectorize it in EVE?

No, BUT we should implement POPTICSS (MOPTICS + MSTTOCLUSTERS) + sOPTICS

### Variants of the algorithm

- MOPTICS, POPTICSS, POPTICSD: https://ieeexplore.ieee.org/document/6877482
- Tra-POPTICS, G-Tra-POPTICS, T-OPTICS: https://www.researchgate.net/publication/277349039_A_scalable_and_fast_OPTICS_for_clustering_trajectory_big_data
- FOPTICS
- sOPTICS (same paper as sDBSCAN)


## Gaussian mixtures

* **Parameters:** many
* **Scalability:** Not scalable
* **Usecase:** Flat geometry, good for density estimation, inductive
* **Geometry:** Mahalanobis distances to centers


### What? How?
#### What is the algorithm trying to find or optimize?
Partition $n$ samples by estimating the parameters of a mixture of $k$ normal distributions to maximize the **Maximum Likelihood** (specifically, the log-likelihood) of the observed data.
Unlike K-Means which uses hard boundaries, it finds "soft" partial memberships (probabilities) for each point across each constituent distribution.
Formally, the objective function the algorithm seeks to maximize is:

$$\arg\max_{\phi, \mu, \Sigma} \sum_{i=1}^{n} \log \left( \sum_{s=1}^{k} \phi_{s} \mathcal{N}(x_{i} | \mu_{s}, \Sigma_{s}) \right)$$

Where $\mathcal{N}$ represents the probability distribution function of a multivariate Gaussian parameterized by mean $\mu_{s}$ and covariance $\Sigma_{s}$, and $\phi_{s}$ represents the mixture weights.

#### Algorithm overview
1. Choose the initial weights, means, and covariance matrices (many methods: K-Means, random data points, etc.).
2. Loop (Expectation-Maximization):
    1. **E-step:** Compute the expectation values for the membership variables (responsibilities) of each data point, determining their probability of belonging to each distribution given the current parameters.
    2. **M-step:** Calculate new plug-in estimates for the distribution parameters (weights, means, and covariances) by performing weighted aggregations over the data points, using the E-step memberships as the weights.

    Stop the loop when the change of the log-likelihood (or lower bound) is less than a predefined convergence threshold.

#### Time complexity?
Per EM iteration:
* Full covariance: $\mathcal{O}(n k d^2 + k d^3)$
* Tied full covariance: $\mathcal{O}(n k d^2 + d^3)$
* Diagonal covariance: $\mathcal{O}(n k d)$
* Spherical covariance: $\mathcal{O}(n k d)$

Total runtime multiplies by the number of EM iterations t.

#### Space complexity?
If responsibilities are materialized, space is $\mathcal{O}(nd + nk + \text{cov params})$.
If responsibilities are streamed and only sufficient statistics are accumulated, working memory can avoid the $\mathcal{O}(nk)$ term.

Covariance parameter storage:
* Full: $\mathcal{O}(k d^2)$
* Tied full: $\mathcal{O}(d^2)$
* Diagonal: $\mathcal{O}(k d)$
* Spherical: $\mathcal{O}(k)$

#### What are the heaviest mathematical operations?
* Compute $k$ probability densities (involving Mahalanobis distances and exponentials) for n vectors at step 2.1.
* Compute $k$ weighted means of in total n vectors at step 2.2.
* Compute $k$ covariance matrices (involving vector outer products) of in total n vectors at step 2.2.
* Compute $k$ matrix inversions or Cholesky decompositions for the covariance matrices at step 2.2.
* Compute the log-likelihood (or lower bound) over n vectors to check if we end the loop (this is usually a computationally cheap byproduct of step 2.1).

#### Data dependency on looping operation?
Between iterations only

#### Nested control flow/branching?
No

#### Irregular Whiles ?
No

#### Reductions necessary?
* Min-reduction and Sum-reduction (if using K-Means) in init. step
* Sum-reduction (over dimensions for dot products, and over clusters for normalization) in step 2.1.
* Sum-reduction (over data points for weighted aggregations) in step 2.2.
* Sum-reduction (for total log-likelihood) and Bool-reduction (for threshold comparison) in convergence check

#### Random Memory access patterns (or Strided)?
No

#### Compute-bound or memory-bound?
hmmmm idk


### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?
No

#### Are there example implementations?
No

#### Or maybe there are GPU versions?
Nah

### Sources

* https://scikit-learn.org/stable/modules/mixture.html
* https://en.wikipedia.org/wiki/Mixture_model
* https://en.wikipedia.org/wiki/Mahalanobis_distance
* https://perception.inrialpes.fr/people/Horaud/Courses/pdf/Horaud-MLSVP8.pdf

### Conclusion: Should we vectorize it in EVE?

Yes. Lots of math, not the easiest alg on this part, but quite epic for vectorization

### Variants of the algorithm

* **Bayesian GMM**: A variant where the parameters and weights are themselves treated as random variables, and prior probability distributions are placed over them. This allows the model to inherently penalize overly complex clusterings and can automatically effectively shut down unused clusters (e.g., Dirichlet Process GMMs).
* **Markov chain Monte Carlo**: As an alternative to the EM algorithm
* **Robust EM**: https://www.sciencedirect.com/science/article/abs/pii/S0031320312002117 No need to know the correct number of clusters in advance, and less sensitive to initial values. But $O(n^2(1+d^2)+n)$ computational complexity for the first iteration...


## BIRCH

* **Parameters:** branching factor, threshold, optional global clusterer.
* **Scalability:** Large n_clusters and n_samples
* **Usecase:** Large dataset, outlier removal, data reduction, inductive
* **Geometry:** Euclidean distance between points


### What? How?
#### What is the algorithm trying to find or optimize?



#### Algorithm overview



#### Time complexity?



#### Space complexity?



#### What are the heaviest mathematical operations?



#### Data dependency on looping operation?



#### Nested control flow/branching?



#### Irregular Whiles ?



#### Reductions necessary?



#### Random Memory access patterns (or Strided)?



#### Compute-bound or memory-bound?




### SIMD versions
#### Are there papers describing SIMD (vectorized!) versions of the algorithm?



#### Are there example implementations?



#### Or maybe there are GPU versions?



### Sources

* https://scikit-learn.org/stable/modules/clustering.html#birch
* https://en.wikipedia.org/wiki/BIRCH
* https://arxiv.org/abs/2006.12881
* https://www.sciencedirect.com/science/article/pii/S0306437921001253

### Conclusion: Should we vectorize it in EVE?



### Variants of the algorithm



## Ranking (from the highest priority to lowest)

1. K-Means
1. EM GMM
1. (s)DBSCAN
1. HDBSCAN
1. Affinity Propagation
1. POPTICSS + sOPTICS
1. Mean-Shift
1. Agglomerative Clustering
1. Spectral Clustering


## Misc stuff
* Maybe https://www.sciencedirect.com/science/article/abs/pii/S0167739X17319271?via%3Dihub would be interesting? Haven't read it yet.
* Look at Density-Peaks Clustering, and its parallel versions: https://neuzhangyf.github.io/assets/pdf/paper-2023-fgcs-density.pdf