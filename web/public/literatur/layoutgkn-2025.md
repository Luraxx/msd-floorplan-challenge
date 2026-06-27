## **LayoutGKN: Graph Similarity Learning of Floor Plans** 

Casper van Engelenburg, Jan van Gemert, Seyran Khademi Delft University of Technology 

Figure 1. **Joint embeddings architectures for graph similarity learning.** ( _**Left**_ ) Graph embeddings networks (GENs) independently embed graphs _G_ 1 and _G_ 2 in a vector space using parameter-shared graph networks (gray box) followed by pooling (orange box). The similarity _s_ is computed based on the vector embeddings. ( _**Middle**_ ) In addition to GENs, graph matching networks (GMNs) model crossgraph node-level interactions (blue arrows) between the two sets of node features in each layer. Embeddings ( _Hi_ ) depend on the other. ( _**Right**_ ) Our graph kernel network (GKN) shifts node-level interactions to the similarity function via a graph kernel (green box), enabling independent embeddings while preserving node-level interactions. 

## **Abstract** 

## **1. Introduction** 

_Floor plans depict building layouts and are often represented as graphs to capture the underlying spatial relationships. Comparison of these graphs is critical for applications like search, clustering, and data visualization. The most successful methods to compare graphs_ i.e _., graph matching networks, rely on costly intermediate cross-graph node-level interactions, therefore being slow in inference time. We introduce_ _**LayoutGKN** , a more efficient approach that postpones the cross-graph node-level interactions to the end of the joint embedding architecture. We do so by using a differentiable graph kernel as a distance function on the final learned node-level embeddings. We show that LayoutGKN computes similarity comparably or better than graph matching networks while significantly increasing the speed. Code and data are open._ 

> To appear in BMVC ’25. Mail: c.c.j.vanengelenburg@tudelft.nl 

In contrast to the natural environment, the built environment follows a remarkably simple spatial logic – which, more remarkably so, seems to be consistent anywhere on earth. When stripped of its details, our built environment can be adequately described as a hierarchy of networks: the main infrastructure as a network of roads connecting cities and towns; cities as a network of roads that divide and bring about the composition of neighborhoods and buildings; buildings as stacks of floor plans; floor plans as networks of rooms and corridors; etc. Often modeled as graphs of nodes and edges, these networks are embedded in physical space, with nodes and edges carrying information about location, shape, and spatial relations. Encoding such relations in machine-readable form has enabled computer-aided methods to understand [21], analyze [9, 22], and synthesize [8, 19] our built environment. In this work, we focus on floor plans and floor plan representation learning. Nonetheless, most ideas could prove worthy in the other ‘levels’ of 

1 

our built environment. 

As visual depictions of building layouts, floor plans offer an elementary yet powerful abstraction of spatial structure—an aspect central to the quality of space [1, 10, 29]. Of key importance to floor plan machine understanding is the ability to effectively compare floor plans based on their spatial structure, as well as the ability to do so _quickly_ – especially for search-related tasks, data visualization, and clustering or classification. The question of spatial similarity is inherently subjective, multi-faceted, and task-dependent. However, we believe, as do others [12, 20, 21, 24], that floor plan similarity can be adequately modeled as a graph comparison problem, given the availability of the graph through image vectorization. 

Most notably, graph matching networks (GMNs) [15] have been successful at ‘solving’ graph comparison problems, and have been adopted for use in floor plan similarity [12, 21]. A GMN is a form of a joint embedding architecture (JEAs), in which two parameter-sharing neural networks ( _i.e_ ., encoders) process two distinct data samples simultaneously to arrive at two corresponding embeddings [4, 5]. The similarity is measured based on a distance ( _e.g_ ., Euclidean) between the embeddings. For GMNs, the encoders are graph neural networks (GNNs). The key distinction between classical JEAs is that, in GMNs, information is exchanged between nodes across both graphs during both training and inference. These cross-graph node-level interactions allow for the explicit modeling of node-level correspondences, which are otherwise thrown away when the graph-level embeddings are computed independently. Although effective, cross-graph node-level interactions do not allow for the offline computation of embeddings for a given database of floorplans, due to the pairwise dependence of the embeddings. Cross-interaction modules hinder the usability of GMNs in real-time retrieval systems where similarity, i.e., the distance between the embeddings, needs to be computed fast. Our contributions enable an effective yet efficient solution for graph-based floor plan retrieval, and are summarized as follows: 

- We propose LayoutGKN: a GNN model built on a differentiable graph similarity metric, which effectively captures the spatial similarity in floor plan representation learning. 

- We model the node-level interactions at the end of the JEA by using a differentiable path-based graph kernel. 

- Experiments on RPLAN and zero-shot generalization to MSD demonstrate that LayoutGKN achieves comparable or better ranking performance than LayoutGMN while being significantly faster. 

## **2. Related works** 

**Floor plan similarity** Floor plan retrieval, cast as a problem of similarity learning, in which the goal is to rank a 

gallery database of floor plans to a query, has been the subject to previous works [12, 13, 20, 21, 24, 26, 30, 31, 34]. Relevant other applications include the retrieval of interface layouts and documents [3, 18, 21, 36] and room layouts [7]. Determining the similarity between two floor plans, as is framed in [21], is a multi-faceted task, regarding semantic ( _e.g_ . room functions), geometric ( _e.g_ ., room shapes), and relational ( _e.g_ ., permeability or adjacency) information layers. The _intersection-over-union_ (IoU) is often considered as an instrumental measure for spatial similarity [12, 18, 21]. While relatively quick to compute, IoU is sensitive to translations, rotations, and scale changes – and, most importantly, does not explicitly measure the relational similarity between the components that compose the layout [31]. Similar to most other works [12, 13, 18, 20, 21, 24, 30, 31], we frame floor plan similarity as a graph comparison problem, in which we model floor plans as spatially-attributed graphs of nodes that represent rooms and edges that define their spatial relations. Unlike previous learning-based methods [12, 18, 21] that train and evaluate under the IoU metric, we do so using a graph-based similarity metric, which we show better aligns with human judgment of floorplan similarity. 

**Graph similarity** To measure similarity between graphs, many works resort to the use of a _graph edit distance_ (GED) [25], which for example is used to assess compatibility in the generation of floor plans [19], or the _maximum common subgraph_ (MCS) used directly for floor plan retrieval [30]. GED is a known NP-complete problem (details in [39]), exponential in the number of nodes, thus hindering the use where similarity needs to be computed fast ( _e.g_ ., in search engines). To address limitations in efficiency, graph kernels (GKs), such as those defined in [6, 27], have been proposed to solve graph comparison or related problems. We encourage the reader to read a recent review on GKs [14]. While graph kernels are effective for comparing relational structures, a key limitation is their reliance on handcrafted input features, such as node attributes. We use graph kernels to compute similarity between floor plans; however, unlike traditional methods, our approach learns the node features directly from data. 

**Graph similarity learning** In turn, learning-based solutions using _graph neural networks_ (GNNs) have been proposed for the problem of comparing graphs [2, 15, 17, 23]. Most notable is the line on _graph matching networks_ (GMNs)[15] that explicitly encodes cross-graph node-level interactions in the joint embedding architecture, which captures fine-grained structural correspondences between the nodes. Benchmarks on floor plan similarity, such as _LayoutGMN_ [21] extensively use GMNs. Problematic to the use of GMNs is the fact that the node/graph embeddings 

2 

cannot be computed in isolation [40], drastically decreasing efficiency in real-time retrieval. Instead of modeling expensive cross-graph node-level interactions _across the GNNs_ , we only model such interactions at the end of the pipeline using a differentiable path-based graph kernel, _GraphHopper_ [6], on top of the final node embeddings. Because the node embeddings can be precomputed independently, our method is much faster than is the case for GMNs (Fig. 1). 

## **3. Problem formulation** 

## **3.1. Floor plans as attributed graphs** 

To explicitly model spatial relations, a floor plan is represented as an undirected graph _G_ = ( _V, E_ ) of nodes _u ∈V_ that describe rooms connected by edges ( _u, v_ ) _∈E ⊆V ×V_ that describe the permeability between the rooms. We should not forget that the use of the graph as representational device of a floor plan is by no means a new idea, and stems, most notably, from a vast line of works in _Space Syntax_ ( _e.g_ ., as in [10] in the mid 70s) and other works around the same time ( _e.g_ ., in Alexander’s [1] and Steadman’s [29] seminal works). In these works, only the graph’s connectivity ( _i.e_ ., its topology) was usually explored. Similar to other works of late [19, 21], we expand the graphs by populating the nodes and edges with a rich set of attributes, which we do as follows. Specifically, each node _u_ is endowed with a package of node features. 1) A one-hot encoding of the category of the room’s function **c**[(] _[u]_[)] _∈_ R[8] . Categories are: ”living room”, ”bedroom”, ”kitchen”, ”bathroom”, ”dining”, ”store room”, ”balcony”, ”corridor”. For example, [1; 0; 0; 0; 0; 0; 0] describes the living room. 2) A vector that characterizes the room’s shape **s**[(] _[u]_[)] _∈_ R[6] , defined as: **s** = [ _cx_ ; _cy_ ; _w_ ; _h_ ; _[√] a_ ; _p/_ 4] where ( _cx, cy_ ) denotes the center of the room, _lx_ the maximum size in _x_ and _ly_ in _y_ , _a_ the area, and _p_ the perimeter. 3) To make use of the graph kernel, we need for each node the shortest-path histogram matrix **M**[(] _[u]_[)] [6]. A path is a sequence of non-repeated nodes connected through edges present in a graph, which for a floor plan captures how one could walk from one space to another. The shortest path between two nodes is the one which traverses the least edges. The entry [ **M**[(] _[u]_[)] ] _ij_ counts how often _u_ occurs at the _i_ -th position on shortest paths of length _j_ . We set the size of **M**[(] _[u]_[)] to R _[δ][×][δ]_ , in which _δ_ is the maximum path length, usually taken as the longest shortest path i.e., the graph diameter. Since graph diameters of floor plan graphs are relatively small, we set _δ_ = 4. Each edge ( _u, v_ ) carries a vector **e**[(] _[u,v]_[)] indicating the permeability of two adjacent spaces: [1; 0] for access connectivity and [0; 1] for adjacent-only. 

## **3.2. Floor plan similarity as graph comparison problem** 

We formulate the challenge of floor plan similarity as a graph comparison problem: We seek a function _s_ , operating on two floor plans represented as attributed graphs _Gi_ and _Gj_ , _s_ ( _Gi, Gj_ ) : _G ×G →_ R[+] , such that, when _Gi_ is similar to _Gj_ , _s_ is large; and when dissimilar, _s_ is small. We narrow down the set of possible solutions by framing the problem in terms of a distance metric learning formulation [38]: We seek a learnable function, e.g., an encoder, _fθ_ , parameterized by _θ_ , that embeds _G_ in a representation space _H_ ( _i.e_ ., _H_ = _fθ_ ( _G_ )). A similarity measure _sH_ is required that computes the similarity between two such representations: 

**==> picture [221 x 12] intentionally omitted <==**

Given the similarity function _sH_ , the objective is to learn _fθ_ ( _G_ ) through a differentiable loss. To decide upon an proper metric for training and evaluation, a user study is conducted, in which we asked architects to rank floor plans. Albeit some level of disagreement, we found a positive and significant correlation between human judgment and a similarity based on a graph edit distance (GED) [25], while almost none for IoU (→ suppl. mat. for details). Node and edge categories are considered when computing GED. Node categories correspond to the room types and edge categories correspond to the type of room-to-room permeability, which can either be access ( _i.e_ ., door) or adjacentonly ( _i.e_ ., wall but no door). We consider the weight of each edit operation ( _e.g_ ., changing node category, deleting / adding edge) the same. Unlike other learning-based frameworks that train and measure the goodness of retrieval based on IoU [12, 18, 21], we train and evaluate based on a normalized variant of GED to express the similarity: sGED( _Gi, Gj_ ) = exp( _−_ GED( _Gi, Gj_ ) _/_ ( _|Gi|_ + _|Gj|_ )), where _|G|_ is the size of the graph ( _i.e_ ., the number of nodes). 

## **4. Method** 

## **4.1. Graph kernel network** 

The joint embedding architecture of our network contains two elements: a graph neural network (GNN) to learn node embeddings, and a graph kernel to compute the similarity between two sets of node embeddings. If _fθ_ denotes the GNN and _sH_ the similarity function, our framework, coined as graph kernel network (GKN), computes the similarity between graphs as given in Eq 1. How are methods fits the others is highlighted in Fig. 1. An overview of our method is given in Fig. 2. 

**Graph neural network for feature learning** Node feature vectors are encoded using separate MLPs: **f** cat : R[8] _→_ 

3 

Figure 2. **Overview of LayoutGKN.** ( _**A. graph extraction**_ ) Semantic images from RPLAN are converted into richly-attributed access graphs. The floor plans’ geometries ( _i.e_ ., the rooms represented as polygons) are centered at (0 _,_ 0) and scaled to fit within the unit square box. The unit box amounts to 20 x 20 meters in reality ( _i.e_ ., 0.1 equals 1 meter). The color indicates the room’s semantic category ( _e.g_ ., dark orange for living room, green for balcony). Edges are modeled when two rooms share a door (black) or a wall (white edge). ( _**B. attributes**_ ) Each node represents a room and is endowed with 3 attributes: the shortest-path matrix, a one-hot encoding of the room’s category and a vector of shape features. ( _**C. training**_ ) LayoutGKN is trained using triplets of graphs, each containing an anchor ( _Ga_ ), positive ( _Ga_ ), and negative ( _Ga_ ) graph. The goal is to penalize the relative distance between anchor-positive and anchor-negative. Anchor, positive, and negative are simultaneously fed into parameter-shared graph neural networks, which consist of a node encoder (dark gray box), followed by a series of _L_ graph message passing network layers (light gray box). This results in embedded graphs ( _Hi_ ). The anchor-positive and anchor-negative similarities are computed as _s_ ap = _kG_ ( _Ha, Hp_ ) and _s_ an = _kG_ ( _Ha, Hn_ ), respectively. The triplet loss _Lt_ penalizes the relative distance log( _s_ an _/s_ ap), given a margin _m_ . 

R _[d]_ on top of the room function vector; **f** shape : R[6] _→_ R _[d]_ on the room shape vector; and **f** edge : R[2] _→_ R _[d]_ on the edge vector. The outputs of **f** cat and **f** shape are concatenated and subsequently fed into another MLP, **f** node : R[2] _[d] →_ R _[d]_ , to form the first hidden state: 

edge encoding **r**[(] _[u,v]_[)] are concatenated and fed into a shared MLP **f** intra : R[3] _[d] →_ R _[d]_ . We use average aggregation over all the individual messages and arrive at the following definition for **m**[(] _l[→][u]_[)] : 

**==> picture [181 x 25] intentionally omitted <==**

To learn expressive node embeddings, we use a message passing network (MPN) similar to the one in [15]. The MPN is a product of _L_ consecutive graph convolutional layers. Each layer contains a message and an update function. The message to node _u_ in layer _l_ , **m**[(] _l[→][u]_[)] , aggregates node information from its direct neighbors _v ∈_ ne( _u_ ) (and itself) and takes into account the type of edges between them. Specifically, for each neighbor _v_ , the hidden node features of the node itself **h**[(] _l[u]_[)] , the one of the neighbor **h** _l_[(] _[v]_[)] , and the 

**==> picture [193 x 43] intentionally omitted <==**

To arrive at the node features at layer _l_ + 1, the aggregated message and the node feature itself are stacked and fed into a GRU, **f** update : R[2] _[d] →_ R _[d]_ : 

**==> picture [178 x 28] intentionally omitted <==**

4 

**Graph kernel as similarity function** We use the GraphHopper path-based graph kernel [6] to compute the similarity between two graphs. Specifically, given two graphs and their corresponding sets of learned node embeddings by the GNN, the graph kernel similarity _kG_ , as shown by the authors, can be computed in a remarkably simple form: as a weighted sum over node kernels across all pairs of nodes and their corresponding vector embeddings: 

negative s.t. 0 _._ 7 _<_ sGED _Ga, Gn_ ) _/_ sGED( _Ga, Gp_ ) _<_ 0 _._ 9. In the case of GKN, during training, we pre-compute the shortest-path histogram matrices. During inference, we precompute the final node embeddings and denominator of _sH_ as well. 

## **5. Results and discussion** 

## **5.1. Experiment setup** 

**==> picture [244 x 36] intentionally omitted <==**

where the node kernel _k_ node ( _•, •_ ) computes the similarity between two nodes, and is typically Gaussian ( _µ ∝ d[−]_[1] [14]): 

**==> picture [224 x 25] intentionally omitted <==**

## **4.2. Training and inference** 

We train LayoutGKN under a triplet network setting [11]; thus, penalizing _relative_ distances. Given a triplet of graphs ( _Ga, Gp, Gn_ ), in which _a_ , _p_ , and _n_ denote anchor, positive, and negative. We feedforward each graph through _fθ_ resulting in embeddings _Ha_ , _Hp_ , and _Hn_ . We use a normalized variant of the graph kernel (Eq. 5) to compute the similarity between two such embeddings: _sH_ ( _Hi, Hj_ ) = _kG_ ( _Hi, Hj_ ) _/_ � _kG_ ( _Hi, Hi_ ) _· kG_ ( _Hj, Hj_ ) _._ Given that we compute the relative distance from anchor-positive to anchor-negative as _d_ apn = _sH_ ( _Ha, Hn_ ) _/sH_ ( _Ha, Hp_ ), we formulate the triplet loss as 

**==> picture [231 x 49] intentionally omitted <==**

where [ _•_ ]+ = max( _•,_ 0). We use the log to effectively penalize small relative distances more. Because the node (Eq. 6) and graph kernel (Eq. 5) are differentiable w.r.t. to the node embeddings, the loss (Eq. 7) is as well. The second line in 7 shows an efficient implementation of our loss in which _kG_ ( _Ha, Ha_ ) is canceled out. Similar to [12, 18, 21], we mine triplets based on the computable metrics we aim to mimic. Our objective, as mentioned before in Sec. 3, is different though: we mimic sGED not MIoU. To find informative triplets, we first rank each floor plan in the training set on MIoU, filter the best 50, and re-rank them on sGED. Each triplet is formed by the query and some combination of positive and negative found in the re-ranked list. Given some anchor, we first pick a positive s.t. 0 _._ 9 _>_ sGED( _Ga, Gp_ ) _>_ 0 _._ 6. To ensure hard negatives, we pick a 

( _**Datasets**_ ) We train and test on RPLAN [35] and evaluate generalization to MSD [32]. After careful cleaning of both datasets, RPLAN and MSD contain 46K+ and 16K+ apartment-level floor plans, respectively. Statistics, descriptions, and pre-processing steps can be found in the suppl. mat. We train our models on RPLAN, and split training and test data with a ratio of 8:2. ( _**Baselines**_ ) We compare LayoutGKN, which we will refer to as GKN in the remainder of the text, with _LayoutGMN_ [21], in short GMN, and basic baseline methods including GEN and GK. GK is the _GraphHopper_ graph kernel as-is, for which we model the node features by concatenating **s**[(] _[u]_[)] and **g**[(] _[u]_[)] . For all methods, we use the same graph representation, as given in Sec. 3. In the case of GEN, GMN, and GKN, the same node encoder and intra-graph message passing mechanisms are used. In the case of GMN, an inter-cross message **mc**[(] _[→][u]_[)] _∈_ R _[d]_ is concatenated to the input of **f** update (Eq. 4), equivalent to [21] (Eq. 4, pp. 4). For GMN and GEN, we use the same graph pooling mechanism as in [15] and use the conventional triplet margin loss on the relative distances between the final graph-level vector embeddings: _Lt_ = [ _d_ an _− d_ an + _m_ ]+. ( _**Evaluation**_ ) For comparing the methods, we report the triplet accuracy, Precision at 5 and 10 (P@5 and 20), and inference times ( _t_ ). We use 4-fold cross-validation, and report the average scores across the folds on the test set. To evaluate the generalization of the methods, we report the zeroshot precision scores on MSD only. ( _**Implementation**_ ) Each model is trained for at most 200 epochs, with early stopping (patience is 10), on an NVIDIA GeForce RTX 4090 Ti GPU using AdamW [16] with a learning rate of 10 _[−]_[4] and batch size of 64. We use layer and batch normalization for the node encoder parts and message passing updates, respectively. 

## **5.2. Effectiveness** 

Table 1 reports, on the left side of the vertical line, the overall effectiveness of all methods. Although similar to GMN in accuracy, GKN outperforms all other methods on ranking. The difference between GKN (or GMN) and GEN scores shows that explicit inclusion of cross-graph nodelevel interactions leads to a significant increase in accuracy and precision. However, we show that modeling expensive cross-graph node-level interactions _across the GNNs_ might not be necessary for effective graph similarity computation. 

5 

Table 1. **Performance comparisons on RPLAN and MSD.** We report: the triplet accuracy; precision (P) scores at 5 and 10; and inference time _t_ per 10K pairs. Best in **bold** . Note that the graph kernel (GK) does not involve any learning: it is GraphHopper kernel [6] as-is, for which we model the node features by concatenating the categorical and shape-based node features ( _i.e_ ., for each node _u_ : [ **c**[(] _[u]_[)] ; **s**[(] _[u]_[)] ]). GK is a strong baseline for retrieval. During inference, embeddings ( _Hi_ ) can only be precomputed for GEN and LayoutGKN, which creates the order of magnitude difference in inference time between these methods and LayoutGMN. * We train and represent the data in LayoutGMN [21] in exactly the same way as is the case for the other methods, which is slightly different from the original implementation in which the triplets are mined using MIoU and floor plans are represented as fully connected graphs. 

|||**RPLAN**|**RPLAN**||**Zero-shot to MSD**|**Zero-shot to MSD**|
|---|---|---|---|---|---|---|
||Accuracy (_↑_)|P@5 (_↑_)|P@10 (_↑_)|_t_[s] (_↓_)|P@5 (_↑_)|P@10 (_↑_)|
|LayoutGK (_baseline_)|65.63_±_0.00|0.389_±_0.000|0.439_±_0.000|1.2_±_0.4|na|na|
|LayoutGEN (_baseline_)|96.24_±_0.07|0.603_±_0.007|0.665_±_0.004|**0.7**_±_**0.1**|0.595_±_0.015|0.605_±_0.018|
|LayoutGMN* CVPR’21|**97.74**_±_**0.05**|0.616_±_0.004|0.675_±_0.002|35.6_±_10.5|0.585_±_0.026|0.596_±_0.020|
|LayoutGKN (_ours_)|**97.78**_±_**0.10**|**0.623**_±_**0.004**|**0.683**_±_**0.002**|1.8_±_0.5|**0.674**_±_**0.024**|**0.697**_±_**0.017**|



Figure 3. ( _**Left and middle**_ ) The effect of the number of learnable parameters on the triplet accuracy. We vary the number of parameters by changing the hidden node dimension and number of graph convolutional layers. The hidden node dimension has a profound effect on the accuracy: the smaller the node dimension, the more a decrease in accuracy. The rate at which the accuracy drops for decreasing node dimension is lowest in the case of GKN. In the case of the number of layers there are negligible differences in accuracy (across all methods). ( _**Right**_ ) Typical training curves (on the validation split) showing that: 1) GKN converges the fastest and 2) has a head-start because of the kernel. 

Instead, which is the case in GKN, the node-level interactions can be ”postponed” to the end, when the final node embeddings are already computed, without losing quality. 

**Effect of size** Fig. 3 shows the effect of network size on triplet accuracy, varying the hidden node dimension ( _d_ ) and number of graph convolutional layers ( _L_ ). When changing _d_ , we fix _L_ = 5; when changing _L_ , we fix _d_ = 64. We observe that performance drops much faster for GEN and GMN than for GKN as _d_ decreases. We attribute this to the different roles of the embeddings. In GEN and GMN, node embeddings must be sufficiently large to capture both (i) their topological role in the graph and (ii) their semantic attributes (e.g., geometry and room function in floor plans). Although GMNs explicitly model cross-graph node interactions, they still depend on expressive node embeddings. In contrast, GKN leverages a topology-aware similarity function ( _i.e_ ., the graph kernel), which naturally accounts for the topology of the graphs when computing the similarity between them. As a result, node embeddings in GKN can be smaller without too much of a performance drop, since 

they need not encode topology as strongly themselves. We have not yet tested with more ‘expressive’ GNN encoders, such as GAT [33] or GIN [37], but the same capacity limitation seems likely to remain. 

**Qualitative studies** Fig. 4 ( **A** ) shows a case of ranking under the sGED metric on RPLAN. A useful property of training floor plan similarity under a sGED metric, is that retrievals are (practically) invariant to grid transformations. In Fig. 4 ( **B** ), the emergence of rotational and flip invariance is depicted. Because of the two first entrees ( _c_[(] _x[u]_[)] and _c_[(] _x[u]_[)][)] of the room shape vector **s**[(] _[u]_[)] , it is important to note that the model is not strictly invariant to such grid transformations. Nonetheless, if you want the model to be _provably_ invariant to such grid transformations, simply remove the first two entrees of **s**[(] _[u]_[)] . 

## **5.3. Efficiency** 

In the case of GMN, we cannot pre-compute the node embeddings, because of the cross-graph node-level interactions. Depending on _d_ , _L_ , and _Ni_ = _|Gi|_ for _i ∈_ (1 _,_ 2), the 

6 

Figure 4. **Examples of ranking on RPLAN.** ( **A** ) Given a query floor plan on the left, the top row shows the ground truth (GT) ranking. The value on top of the floor plan indicates the similarity in terms of sGED. Since the GT scores are quite often equivalent for different retrievals, we color-coded them: the same color means the same GT score. Rows 2 to 4 depict the rankings based on GEN, GMN, and GKN. On top of the floor plans, we provide the GT score and indicate using the same color scheme as is the case in the top row where a floor plan would have landed in the GT ranking. A gray background means that it would fall outside the top-10 on GT. ( **B** ) We show an example of ranking on GKN in which the top-10 retrievals show signs that the retrieval results are invariant to the elementary grid transformation (flips and 90 degree rotations). 

number of FLOPs is order of magnitudes larger for GMNs: _≈_ 5-10k FLOPs in the case of GKN and GEN; _≈_ 10-20M FLOPs in the case of GMN. Therefore, the differences in inference time _t_ , shown on the direct left of the vertical line in Tab. 1, are not surprising: GMN takes _≈_ 20X more time than the rest. In real-time search engines, speed is of great importance: results should pop-up in seconds, not minutes. GMNs compute the similarity between 10K pairs in about 30-40 seconds. GKNs, on the other hand, compute the similarity between 10K pairs in about 1 to 2 seconds: a drastic speedup compared to GMNs – a speedup that, most importantly, aligns with what we are after in real-time search. 

## **5.4. Generalization** 

We further assess the models in a zero-shot setting ( _i.e_ ., no additional training or fine-tuning) by evaluating directly on the more complex MSD dataset [32]. As shown on the right side of Tab. 1, all methods retain relatively high precision despite the distribution shift from RPLAN to MSD. For GKN, precision is even higher on MSD than on RPLAN, and it achieves the strongest performance overall. A plausible explanation is that MSD, which contains entire building complexes, exhibits a higher proportion of near-duplicate or structurally similar floor plans, thereby boosting retrieval scores. The advantage of GKN compared to the other 

methods may further stem from the use of the graph kernel, which provides a strong structural prior aligned with the GED-based evaluation and is thus less dependent on dataset-specific differences. These results indicate stronger zero-shot transfer for GKN, though disentangling intrinsic model robustness from dataset effects remains an open question. 

## **6. Conclusion** 

We introduced LayoutGKN, a graph-based joint embedding architecture for measuring the spatial similarity between floor plan graphs. Unlike graph matching networks (GMNs), LayoutGKN decouples representation learning from similarity computation: node embeddings are learned independently, while important cross-graph node-level interactions are only imposed in the distance function via a differentiable path-based kernel. Our results on floor plan similarity learning indicate that cross-graph node-level interactions are not necessarily required within the graph encoders themselves, but can be effectively postponed to the similarity-based loss: compared to LayoutGMN, our method achieves comparable or better performance on floor plan retrieval on RPLAN, while drastically increasing the speed. We are curious whether our method generalizes to other graph domains ( _e.g_ ., molecules) as well. 

7 

**Acknowledgments** We thank all participants of our user study for their valuable time and feedback. 

## **References** 

- [1] Christopher Alexander, Ishikawa Sara, and Silverstein Murray. _A Pattern Language: Towns, Buildings, Construction_ . Oxford University Press, 1977. 2, 3 

- [2] Yunsheng Bai, Hao Ding, Song Bian, Ting Chen, Yizhou Sun, and Wei Wang. SimGNN: A Neural Network Approach to Fast Graph Similarity Computation. In _Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining_ , 2019. 2 

- [3] Yue Bai, Dipu Manandhar, Zhaowen Wang, John Collomosse, and Yun Fu. Layout Representation Learning with Spatial and Structural Hierarchies. In _AAAI_ , 2023. 2 

- [4] Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard S¨ackinger, and Roopak Shah. Signature Verification using a ”Siamese” Time Delay Neural Network. In _NeurIPS_ , 1993. 2 

- [5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. _ICML_ , 2020. 2 

- [6] Aasa Feragen, Niklas Kasenburg, Jens Petersen, Marleen de Bruijne, and Karsten Borgwardt. Scalable Kernels for Graphs with Continuous Attributes. In _NeurIPS_ , 2013. 2, 3, 5, 6 

- [7] Matthew Fisher, Manolis Savva, and Pat Hanrahan. Characterizing Structural Relationships in Scenes Using Graph Kernels. In _ACM SIGGRAPH_ , 2011. 2 

- [8] Liu He and Daniel Aliaga. COHO: Context-Sensitive CityScale Hierarchical Urban Layout Generation. In _ECCV_ , 2024. 1 

- [9] Congrui Hetang, Haoru Xue, Cindy Le, Tianwei Yue, Wenping Wang, and Yihui He. Segment Anything Model for Road Network Graph Extraction. In _CVPRw_ , 2024. 1 

- [10] B Hillier, A Leaman, P Stansall, and M Bedford. Space Syntax. _Environment and Planning B: Planning and Design_ , 1976. 2, 3 

- [11] Elad Hoffer and Nir Ailon. Deep Metric Learning Using Triplet Network. In _Similarity-Based Pattern Recognition_ , 2015. 5 

- [12] Jiongchao Jin, Zhou Xue, and Biao Leng. SHRAG: Semantic Hierarchical Graph for Floorplan Representation. In _International Conference on 3D Vision_ , 2022. 2, 3, 5 

- [13] Rasika Khade, Krupa Jariwala, and Chiranjoy Chattopadhyay. An Interactive Floor Plan Image Retrieval Framework Based on Structural Features. _Arabian Journal for Science and Engineering_ , 2023. 2 

- [14] Nils M. Kriege, Fredrik D. Johansson, and Christopher Morris. A Survey on Graph Kernels. _Applied Network Science_ , 2020. 2, 5 

- [15] Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, and Pushmeet Kohli. Graph Matching Networks for Learning the Similarity of Graph Structured Objects. In _ICML_ , 2019. arXiv:1904.12787. 2, 4, 5 

- [16] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In _ICLR_ , 2019. 5 

- [17] Guixiang Ma, Nesreen K. Ahmed, Theodore L. Willke, and Philip S. Yu. Deep Graph Similarity Learning: A Survey, 2020. arXiv:1912.11615. 2 

- [18] Dipu Manandhar, Dan Ruta, and John Collomosse. Learning Structural Similarity of User Interface Layouts Using Graph Networks. In _ECCV_ , 2020. 2, 3, 5 

- [19] Nelson Nauata, Kai-Hung Chang, Chin-Yi Cheng, Greg Mori, and Yasutaka Furukawa. House-GAN: Relational Generative Adversarial Networks for Graph-Constrained House Layout Generation. In _ECCV_ , 2020. 1, 2, 3 

- [20] Hyejin Park, Hyegyo Suh, Jaeil Kim, and Seungyeon Choo. Floor Plan Recommendation System using Graph Neural Network with Spatial Relationship Dataset. _Journal of Building Engineering_ , 2023. 2 

- [21] Akshay Gadi Patil, Manyi Li, Matthew Fisher, Manolis Savva, and Hao Zhang. LayoutGMN: Neural Graph Matching for Structural Layout Similarity. In _CVPR_ , 2021. 1, 2, 3, 5, 6 

- [22] Pablo N. Pizarro, Nancy Hitschfeld, Ivan Sipiran, and Jose M. Saavedra. Automatic Floor Plan Analysis and Recognition. _Automation in Construction_ , 140, 2022. 1 

- [23] Pau Riba, Andreas Fischer, Josep Llad´os, and Alicia Forn´es. Learning Graph Edit Distance by Graph Neural Networks. _Pattern Recognition_ , 2020. 2 

- [24] Qamer Uddin Sabri, Johannes Bayer, Viktor Ayzenshtadt, Syed Saqib Bukhari, Klaus-Dieter Althoff, and Andreas Dengel. Semantic Pattern-based Retrieval of Architectural Floor Plans with Case-based and Graph-based Searching Techniques and their Evaluation and Visualization. In _Proceedings of the 6th International Conference on Pattern Recognition Applications and Methods_ , 2017. 2 

- [25] Alberto Sanfeliu and King-Sun Fu. A Distance Measure Between Attributed Relational Graphs for Pattern Recognition. _IEEE Transactions on Systems, Man, and Cybernetics_ , 13, 1983. 2, 3, 11 

- [26] Divya Sharma and Chiranjoy Chattopadhyay. High-level Feature Aggregation for Fine-Grained Architectural Floor Plan Retrieval. _IET Computer Vision_ , 2018. 2 

- [27] Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler-Lehman Graph Kernels. _Journal of Machine Learning Research_ , 2011. 2 

- [28] Matthias Standfest, Michael Franzen, Yvonne Schr¨oder, Luis Gonzalez Medina, Yarilo Villanueva Hernandez, Jan Hendrik Buck, Yen-Ling Tan, Milena Niedzwiecka, and Rachele Colmegna. Swiss Dwellings: A Large Dataset of Apartment Models Including Aggregated Geolocation-based Simulation Results Covering Viewshed, Natural Light, Traffic Noise, Centrality and Geometric Analysis. https: //zenodo.org/records/7788422, 2022. 10 

- [29] Philip Steadman. _Architectural Morphology: An Introduction to the Geometry of Building Plans_ . Pion, 1983. 2, 3 

- [30] Yuki Takada, Naoto Inoue, Toshihiko Yamasaki, and Kiyoharu Aizawa. Similar Floor Plan Retrieval featuring MultiTask Learning of Layout Type Classification and Room Presence Prediction. In _ICCE_ , 2018. 2 

8 

- [31] Casper van Engelenburg, Seyran Khademi, and Jan van Gemert. SSIG: A Visually-Guided Graph Edit Distance for Floor Plan Similarity. In _ICCVw_ , 2023. 2, 10 

- [32] Casper van Engelenburg, Fatemeh Mostafavi, Jan van Gemert, and Seyran Khademi. MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes. In _ECCV_ , 2024. 5, 7, 10 

- [33] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua Bengio. Graph Attention Networks. In _ICLR_ , 2018. 6 

- [34] Raoul Wessel, Ina Bl¨umel, and Reinhard Klein. The Room Connectivity Graph: Shape Retrieval in the Architectural Domain. In _The 16-th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision_ , 2008. 2 

- [35] Wenming Wu, Xiao-Ming Fu, Rui Tang, Yuhan Wang, YuHao Qi, and Ligang Liu. Data-driven Interior Plan Generation for Residential Buildings. _ACM ToG_ , 38, 2019. 5, 10 

- [36] Xingjiao Wu, Luwei Xiao, Xiangcheng Du, Yingbin Zheng, Xin Li, Tianlong Ma, and Liang He. Cross-Domain Document Layout Analysis via Unsupervised Document Style Guide. _Expert Syst. Appl._ , 2022. 2 

- [37] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural Networks? In _ICLR_ , 2019. 6 

- [38] Liu Yang and Rong Jin. Distance Metric Learning: A Comprehensive Survey. https://www.cs.cmu.edu/ ˜[liuy/frame_survey_v2.pdf][, 2006.][3] 

- [39] Zhiping Zeng, Anthony K. H. Tung, Jianyong Wang, Jianhua Feng, and Lizhu Zhou. Comparing Stars: On approximating Graph Edit Distance. _Proc. VLDB Endow._ , 2009. 2 

- [40] Haoran Zheng, Jieming Shi, and Renchi Yang. GraSP: Simple yet Effective Graph Similarity Predictions. In _AAAI_ , 2025. 3 

9 

## **Supplementary materials** 

Figure 5. **Floor plan datasets and representation.** 

## **Data** 

**Datasets: RPLAN and MSD** We use RPLAN [35] and MSD [32]. RPLAN contains 88K+ floor plans and covers Asian residential apartments. As shown by [31], RPLAN contains a substantial amount of near-duplicates as well as floor plans that are _not_ entirely connected ( _i.e_ ., floor plans that contain rooms or room constellations that are disconnected from the rest). We remove near-duplicates as well as unconnected plans. The number of remaining floor plans is 46K+. MSD originates from [28] and contains 6K+ floor plans of residential building complexes found in Switzerland. We extract the apartmentlevel floor plans by cutting the corresponding floor plan graphs along the entrances to the public spaces. The resulting number of floor plans is 16K+. 

**Homogenization** We extract the graphs from RPLAN ourselves and re-model the ones from MSD. We do so such that the graph formats are homogeneous: floor plans are centered around the origin, the same scale is used (one unit of measurement amounts to 10 meters in reality), and the node and edge features are the same size and contain the same information, in the same order. Each floor plan is modeled as a graph _G_ = ( _V, E_ ) of nodes _u ∈V_ that correspond to the rooms and edges ( _u, v_ ) _∈E_ = _V ⊆V_ connecting the nodes and that indicate permeability of the rooms. Each node _u_ is endowed with a package of node features: a category for room’s function _c_[(] _[u]_[)] _∈_ N[8] ; a vector that characterizes the room’s shape **s**[(] _[u]_[)] _∈_ R[6] . The room shape vector is defined as follows: **s**[(] _[u]_[)] = [ _c_[(] _x[u]_[)] _[, c]_[(] _y[u]_[)] _[, w]_[(] _[u]_[)] _[, h]_[(] _[u]_[)] _[,] √a_[(] _[u]_[)] _,[p]_[(] 4 _[u]_[)][]] _[⊤]_[.][Here,][ (] _[c][x][, c][y]_[)][ denotes the center] of the room, _lx_ the maximum size in _x_ and _ly_ in _y_ , _a_ the area, and _p_ the perimeter. Each edge ( _u, v_ ) carries a vector **e**[(] _[u,v]_[)] indicating the permeability of two adjacent spaces: [1; 0] for access connectivity and [0; 1] for adjacent-only. A visual clarification and some additional statistics are provided in Fig. 5. 

## **Intersection over union** 

We define the _mean intersection over union_ (MIoU) between two floor plans _Xi_ and _Xj_ as follows. For each room category _c_ , let _rc_ ( _X_ ) denote the region of _X_ occupied by class _c_ (either as a set of pixels in a semantic mask or as the union of all polygons with label _c_ ). We consider the set of categories _C∗_ = _{ c | µ_ ( _rc_ ( _Xi_ ) _∪ rc_ ( _Xj_ )) _>_ 0 _}_ , i.e. all classes that occur in at least one of the two plans, where _µ_ ( _·_ ) denotes area (pixel count or polygon area). The MIoU is then given by 

**==> picture [351 x 28] intentionally omitted <==**

10 

## **Graph edit distance** 

The _graph edit distance_ (GED) [25] between two floor plan graphs _Gi_ and _Gj_ is defined as the minimum number of edit operations required to transform _Gi_ into _Gj_ , where operations include inserting, deleting, or relabeling nodes and edges (for example, changing a node label from “living room” to “kitchen”). We convert GED into a normalized similarity score by scaling with the total number of nodes and applying an exponential decay: 

**==> picture [342 x 25] intentionally omitted <==**

where _|G|_ denotes the number of nodes in graph _G_ . 

## **Precision** 

P@k measures the fraction of relevant items among the top- _k_ retrieved results: 

**==> picture [315 x 30] intentionally omitted <==**

where _ri_ is the item at rank _i_ and **1** _{·}_ is the indicator function. In practice, we take the top-50 results returned by the model for each query. We then re-rank them according to the ground-truth similarity values. Finally, we compute P@k between this ground-truth order and the model’s original ranking. 

## **User study** 

We conducted a user study to check how trained architects would rank floor plans on spatial similarity. We randomly sampled 50 floor plan queries and gathered for each floor plan, the top-10 retrievals on MIoU (Eq. 8). The top 10 retrievals are randomly shuffled and presented to the architects. The architects are asked to rank the shuffled floor plans on similarity. We used _Miro_[1] for getting the annotations: an online software platform which made it possible to easily drag and drop the floor plans. A snapshot of the setup is given in Fig. 6. We investigate the correlation between human judgment and the computable metrics ( _i.e_ ., MIoU and sGED). In Fig. 7, we show how the computable metrics evolve for the top-10 floor plans, when ranked by the architects. The horizontal axis indicates the index of the floor plan where it is placed according to the architects: from most (index = 0) to least (index = 9). The vertical axis indicates the score of the computable metric. The first row of plots show the individual curves; the second the mean scores. Clearly, there is very prominent correlation between human judgment and a similarity based on GED ( _i.e_ . sGED). This means that, in general, architects tend to rank floor plans (in the case that they are already quite similar in terms of overall shape) more on connectivity / topology than on the overlap of spaces. Therefore, our objective is to mimic sGED in the representation space. 

## **Hyperparameters** 

Across all methods, we perform a hyperparameter analysis – and report the ones with the highest scores. The hyperparameters we test against are: the learning rate (1 _e[−]_[5] _· · ·_ 1 _e[−]_[3] ); the batch size _{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ ; the node-, edge-, and graph-level hidden dimensions _{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ ; the number of graph convolutional layers _{_ 2 _,_ 3 _,_ 4 _,_ 5 _}_ ; and the triplet margin _{_ 5 _e[−]_[4] _,_ 1 _e[−]_[4] _, · · · ,_ 5 _e[−]_[1] _,_ 1 _}_ . 

> 1https://miro.com/index/ 

11 

**==> picture [142 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
Figure 6. Setup of user study in  Miro .<br>**----- End of picture text -----**<br>


12 

Figure 7. **User study results.** 

13 

