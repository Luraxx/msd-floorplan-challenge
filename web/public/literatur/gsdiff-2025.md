# **GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation** 

**Sizhe Hu**[1*] **, Wenming Wu**[2†‡] **, Yuntao Wang**[3§] **, Benzhu Xu**[4¶] **, Liping Zheng**[5||] 

1Hefei University of Technology 485 Danxia Lu, Shushan District, Hefei City, Anhui Province, 230061, China 

## **Abstract** 

Automating architectural floorplan design is vital for housing and interior design, offering a faster, cost-effective alternative to manual sketches by architects. However, existing methods, including rule-based and learning-based approaches, face challenges in design complexity and constrained generation with extensive post-processing, and tend to obvious geometric inconsistencies such as misalignment, overlap, and gaps. In this work, we propose a novel generative framework for vector floorplan design via structural graph generation, called GSDiff, focusing on wall junction generation and wall segment prediction to capture both geometric and semantic aspects of structural graphs. To improve the geometric rationality of generated structural graphs, we propose two innovative geometry enhancement methods. In wall junction generation, we propose a novel alignment loss function to improve geometric consistency. In wall segment prediction, we propose a random self-supervision method to enhance the model’s perception of the overall geometric structure, thereby promoting the generation of reasonable geometric structures. Employing the diffusion model and the Transformer model, as well as the geometry enhancement strategies, our framework can generate wall junctions, wall segments and room polygons with structural and semantic information, resulting in structural graphs that accurately represent floorplans. Extensive experiments show that the proposed method surpasses existing techniques, enabling free generation and constrained generation, marking a shift towards structure generation in architectural design. Code and data are available at https://github.com/SizheHu/GSDiff. 

## **Introduction** 

Automatic design of architectural floorplans has garnered widespread attention, as detailed architectural blueprints are crucial for constructing residences and designing interior scenes. Recent years have seen significant advancements in the automated floorplan generation. Existing methods can be 

- *https://orcid.org/0000-0001-9308-5768 

- †Corresponding author: wwming@hfut.edu.cn (Wenming Wu) 

- ‡https://orcid.org/0000-0002-0640-8520 

- §https://orcid.org/0009-0003-2827-5727 

- ¶https://orcid.org/0000-0001-7756-0901 

- ||https://orcid.org/0000-0001-5071-9628 

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

broadly categorized into rule-based and learning-based approaches. The former (Merrell, Schkufza, and Koltun 2010; Liu et al. 2013; Laignel et al. 2021; Shekhawat et al. 2021), relying on specific user requirements and expert knowledge, typically optimize based on various explicit rules as constraints. This process is often sensitive to the modeling of constraints and the selection of parameters. The latter (Hu et al. 2020; Chaillou 2020; Nauata et al. 2021; Para et al. 2021), using deep neural networks, learns implicit design rules from real floorplans. While addressing the shortcomings of rule-based methods, it also introduces new issues: (i) It is challenging to ensure that generated floorplans meet explicit constraints, such as pronounced misalignment; (ii) The generated results usually require heuristic post-processing to be converted into usable vector floorplans. 

In this paper, we propose a novel framework called _GSDiff_ to directly synthesize vector floorplans. The core idea is to view vector floorplan synthesis as structural graph generation and decouple it into two tasks: wall junction generation and wall segment prediction. We represent the floorplan as a structural graph (Sun et al. 2022), where nodes represent wall junctions and edges represent wall segments. Additionally, to capture the floorplan semantics, we consider room labels as one of the node attributes. We first use a generative model based on a diffusion model to generate graph nodes, and then a predictive model based on a Transformer is used to determine graph edges between generated nodes, resulting in a complete structural graph. To improve the design aesthetics, we also propose geometry-enhanced optimization techniques. During the node generation phase, we introduce a novel node alignment loss that optimizes the alignment error of nodes in mixed-base representations, which empowers the generative model to constrain node alignment. In the edge prediction phase, we employ an innovative edge perception enhancement strategy. This involves randomly interpolating a third point on the edges and self-supervising the model to predict the interpolation coefficients, which enhances the geometric perception ability of our edge prediction model, thereby improving the topological connectivity of structural graphs. Finally, vector floorplans can be directly extracted from the generated structural graphs. 

Extensive evaluations show that our method has significant advantages over state-of-the-art techniques on all metrics, enabling free generation and constrained generation. 

Our contributions are as follows: (i) A novel framework for automatically generating diverse, high-quality vector floorplans with various constraints by transforming the problem into a structured graph generation process. (ii) An alignment error optimization strategy that improves node alignment for better node consistency in the node generation phase. (iii) An innovative edge perception enhancement strategy that improves the edge accuracy in the edge prediction phase. 

## **Related work** 

Early works generate floorplans through rule-based floorplan optimization (Merrell, Schkufza, and Koltun 2010; Liu et al. 2013; Wu et al. 2018; Laignel et al. 2021; Wang and Zhang 2020; Shekhawat et al. 2021; Bisht et al. 2022). Due to the complexity of architectural design, deep learning methods have now become the mainstay of the field. Therefore, we focus on learning-based floorplan generation. 

**Imagery floorplan generation** Some deep learning methods generate imagery floorplans. _RPLAN_ (Wu et al. 2019) proposes a two-stage method for floorplan generation that starts with predicting room locations and types, followed by detailing wall semantics and finalizing with vectorization to achieve the end floorplan. _WallPlan_ (Sun et al. 2022) also converts floorplan generation as a graph generation task. However, it still generates floorplan images rather than vector formats via Convolutional Neural Networks (CNNs), therefore it can be categorized as an imagery generation. These methods cannot operate as true generative models as they generate specific outputs from given inputs. GANbased floorplan generation (Chaillou 2020; Nauata et al. 2020, 2021) have gained traction for this purpose. However, these methods face challenges in generating structural elements, requiring complex post-processing to convert to vector formats. In contrast, our method not only can generate multiple results from the same input but also bypasses such limitations by directly generating vector floorplans, simplifying the process and enhancing output usability. 

**Vector floorplan generation** Vector floorplans are more widely used in practical applications. _Graph2Plan_ (Hu et al. 2020) introduces a method to create floorplans from specified bubble diagrams. However, aligning the boxes with the semantic representations requires complex post-processing to obtain vector floorplans. (Para et al. 2021) conceptualizes the floorplan as a box set, followed by optimization for geometric shaping. This method sometimes faces unsuitable constraints for optimization. _HouseDiffusion_ (Shabani, Hosseini, and Furukawa 2023) represents an innovative application of diffusion models to floorplan generation, where floorplans are depicted as polygons with vertices categorizing rooms or doors. This method, however, encounters issues with room alignment, producing gaps or overlaps, and is limited by the necessity of specifying room categories and numbers up front. Our work distinguishes itself by avoiding the limitations of box-set representations and directly producing vector floorplans. Furthermore, by utilizing a generative model, our method possesses the ability to generate diverse results from the same input, while Graph2Plan can only produce a single output. 

**Diffusion Model** Diffusion models are a class of generative models to reverse the noise addition process, thereby enabling the generation of data that mimics the original distribution from Gaussian noise. A standard diffusion model generally contains the forward and backward processes (Ho, Jain, and Abbeel 2020). Diffusion models have made remarkable progress across various generation tasks, including image generation (Nichol et al. 2021; Ho et al. 2022; Rombach et al. 2022; Saharia et al. 2022), point cloud generation (Nichol et al. 2022), and 3D model generation (Poole et al. 2022; Liu et al. 2023). Our framework is centered on the diffusion model, and extends the capabilities of diffusion modeling for generating floorplans, enhancing the flexibility of the model to generate a wider variety of floorplans without the need for predefined room categories or quantities. 

## **Method** 

We convert floorplan generation into graph generation, by representing the vector floorplan as a structural graph _G_ = ( _V, E_ ). Directly generating structural graphs _P_ ( _G_ ) is pretty complex. To this end, we propose a novel generation framework _GSDiff_ (Figure 1) to decouple the structure graph generation into two stages: node generation and edge prediction, i.e., _P_ ( _G_ ) = _P_ ( _V, E_ ) = _P_ ( _V_ ) _P_ ( _E|V_ ), which results in the complete structural graph. Finally, all minimal polygonal loops of the structural graph are extracted as rooms to obtain the final vector floorplan. _GSDiff_ takes design constraints, such as floorplan boundaries, as input and produces high-quality vector floorplans. For simplicity, we first introduce unconstrained generation and more details about constrained generation will be presented later. 

## **Floorplan representation** 

We represent the vector floorplan as a structural graph (Figure 1 (a)): wall junctions are abstracted as nodes, and wall segments as edges. _G_ = ( _V, E_ ), where _V_ = _{v_ 1 _, . . . , vn}_ is the node set, and _E_ = _{_ ( _vi, vj_ ) _|vi, vj ∈ V }_ is the edge set. _vi_ = ( _ci, ri, bi_ ), where _ci_ = ( _xi, yi_ ) _∈_ R[2] is the positional coordinate in the range of [ _−_ 1 _,_ 1). _ri ∈_ [0 _,_ 1] _[n]_ denotes the semantics of all rooms that surround _vi_ , where 0 for absence and 1 for presence of the specific room category, and _n_ is the number of room categories, which is 7 ( _Living room_ , _Bedroom_ , _Kitchen_ , _Bathroom_ , _Balcony_ , _Storage_ , _External area_ ) in our experiments. Let _F_ = _{R_ 1 _, R_ 2 _, . . . , Rm}_ represent all polygonal loops of rooms, each loop _Ri_ = _{vi,_ 1 _, vi,_ 2 _, . . . , vi,Ni|vi,j ∈ V }_ is defined by a sequence of graph nodes. The room category of _Ri_ is the shared room category of all graph nodes forming that room. Different from (Sun et al. 2022), we set a fixed node number _N_ = 53 by considering background nodes, which is determined by the maximum number of nodes in floorplans in our dataset. Background nodes are filtered out by assigning a background attribute for each node, denoted as _bi ∈{_ 0 _,_ 1 _}_ , where 0 denotes the junction node and 1 for the background node. Vector floorplans have a multi-level structure. Spatially, 0D wall junctions form 1D wall segments, which close into 2D rooms. Semantically, each wall junction that forms a room has the same room semantics. Therefore, generating 

**==> picture [499 x 316] intentionally omitted <==**

**----- Start of picture text -----**<br>
1000 100 10 0<br>Time Og<br>- -<br>Bedroom Bathroom<br>v  = ( c, r, b )<br>c  : ( x, y ) Living room<br>r  : (Bedroom ,  Bathroom ,  Living room) Bedroom<br>b  : Wall junction Kitchen<br>e Bathroom<br>Living room<br>Balcony<br>e Storage<br>(a) Floorplan representation (b) Generation process<br>Figure 1: Overview. We represent the floorplan as a structural graph (a) and transform the vector floorplan synthesis into struc-<br>tural graph generation (b). We first generate graph nodes using a diffusion model, then predict the existence of edges between<br>each pair of nodes, and finally extract rooms represented by polygons with semantic labels, resulting in vector floorplans.<br>Node generation Forward process Edge prediciton Floorplan extraction<br>VT a-7 Reverse process Bee TT Vt H Vt -1 ee Reverse process TTS Sy V0 i poop t √ ood: [          ][   ] ) [      ] [          ][   ] ° =e<br>Edge Transformer √<br>×<br>a ' iC } o— —+ x<br>i HE H<br>a<br>i: peseeeeeeesnnnnccceceeeesnnnncenceeeetnnnnggeceeeeeesnnsceeeeeeeetnnnnneeeeeeessnnnnseeeeeetegnnnseeeeeetey<br>11 iO:ide 11 iia Coordinate embedding io} Endpoint Living room e Bathroom ::<br>Node Transformer Semantics-background embedding Interpolation point Bedroom Balcony<br>0 f ie LE ' in Time step embedding + Add C) Kitchen e Storage :<br>4 i m oo : ' id Void embedding ® Concatenation i<br>                t Vt Vt -1 Fused embedding<br>**----- End of picture text -----**<br>


Figure 1: Overview. We represent the floorplan as a structural graph (a) and transform the vector floorplan synthesis into structural graph generation (b). We first generate graph nodes using a diffusion model, then predict the existence of edges between each pair of nodes, and finally extract rooms represented by polygons with semantic labels, resulting in vector floorplans. 

Figure 2: Network architecture of _GSDiff_ . we propose to decouple the structure graph generation into two stages: node generation and edge prediction, which results in the complete structural graph. Finally, all minimal polygonal loops of the structural graph are extracted as rooms to obtain the final vector floorplan. 

all 0D wall junctions basically determines the whole floorplan structure. 

## **Node Generation** 

We adopt a diffusion model (Ho, Jain, and Abbeel 2020) based network architecture to generate nodes (Figure 2). 

**Network architecture** Our forward process incrementally adds noise to an original data sample _V_ 0 over _T_ steps, resulting in a sample resembling Gaussian noise. Our reverse process is a Transformer (Vaswani et al. 2017) based neural network, which takes the noisy node set at time _t_ and outputs the predicted noise _ϵθ_ at time _t −_ 1 ( _θ_ represents the parameters of Transformer), therefore inferring the noisy node set at time _t −_ 1. When the time step reaches 0, the node generation process terminates. Given a noisy node set _Vt_ at time _t_ . We initialize the node embedding of node _fi_ as 

**==> picture [164 x 14] intentionally omitted <==**

_fi_[c][=][[] _[γ]_[(] _[x][i]_[)] _[, γ]_[(] _[y][i]_[)]] _[∈]_[R] _[d]_[is][the][coordinate][embedding,] where _γ_ ( _·_ ) is the positional encoding (Vaswani et al. 2017). _fi_[(] _[r,b]_[)] _∈_ R _[d]_ is the embedding of ( _ri, bi_ ) using a fully connected layer. _fi[t][∈]_[R] _[d]_[is][the][time][embedding][using][a][feed-] forward neural network (FFN). 

**Loss function** The reconstruction loss is typically defined as the Mean Squared Error (MSE) for training: 

**==> picture [193 x 12] intentionally omitted <==**

Due to the probabilistic nature of neural networks, the generated nodes may not be perfectly aligned. We propose a new alignment loss that optimizes alignment errors. Unlike natural language or images, structural graphs have precise geometric relationships, like perfectly horizontal or vertical walls. Directly regressing real-valued coordinates often fails to capture this precision. Thus, we convert real values into binary representations for regression, which discretizes the continuous coordinate space for more precise learning. However, learning discrete representations accurately is challenging, as errors in higher-order bits can cause significant real-value errors. To address this, we propose mixing multiple radix representations. Real-valued representations can heavily penalize large errors but are lenient on small misalignments. Binary representations, though overly discrete and causing significant penalties for small misalignments, are ineffective for large errors. By “interpolating” various radix representations, we aim for a smooth transition that penalizes large errors appropriately while remaining sensitive to small misalignments, balancing both advan- 

tages for better performance. We aim to enhance node alignment by applying the above concepts to propose a novel alignment loss across multiple bases, including real, binary, quaternary, octal, and hexadecimal: 

**==> picture [215 x 24] intentionally omitted <==**

**==> picture [232 x 77] intentionally omitted <==**

where _Base[k]_ ( _·_ ) is the _k_ -base representation, ∆ _b[∗] i_ = min _j_ = _i |b[∗] i[−][b][∗] j[|]_[,] _∈{X, Y }_ , _g_ ( _x_ ) = _−_ 2 _∗_ log 1 _−[x]_ 2[,] _g[k]_ ( _x_ ) = _−d[k]_ log 1 _− d[x][k]_[,][with] _[d][k]_[indicating][the][maxi-] mum allowable di nce u er _k_ -base. _n_ is the node number, _s_ is the bit size. With _k_ = 2 _,_ 4 _,_ 8 _,_ 16, _s_ = 12 _,_ 6 _,_ 5 _,_ 3 and _d[k]_ = 12 _,_ 18 _,_ 35 _,_ 45. ( _·_ ) _i_ denotes the _i_ -th bit. 

We combine the reconstruction loss and alignment loss: 

**==> picture [197 x 13] intentionally omitted <==**

We adopt time-related weighting scheme _ω_ ( _t_ ) (Chen et al. 2024), assigning higher weights at smaller time step _t_ . More details can be found in the supplementary material. 

## **Edge Prediction** 

We use a Transformer-based predictive model to determine graph edges between generated nodes (Figure 2). 

**Network architecture** For each candidate edge ( _vi, vj_ ), the input embedding is obtained by fusing the embeddings of _vi_ and _vj_ : 

**==> picture [149 x 11] intentionally omitted <==**

_d_ The coordinate embedding _fi[c] ∈_ R 2 and the semantic- _d_ background embedding _fi_[(] _[r,b]_[)] _∈_ R 2 are concatenated as the node embedding _fi_ = _fi[c][⊕][f] i_[ (] _[r,b]_[)] . To enhance the model’s robustness, we introduce noise to the node features during the training phase. Specifically, for the normalized 2D coordinate of each node, we add truncated Gaussian noise _ϵ[′] ∼ Truncate N_ (0 _, σc_[2][)] _[,][ −]_[3] _[σ][c][,]_[ 3] _[σ][c]_[,][which][sampled] from Gaussian no e but is bounded at oth ends. For the semantic attributes of the nodes, we randomly flip each bit in the multi-hot representation with a probability _pflip_ , simulating label noise. We set _σc_ = 1 and _pflip_ = 0 _._ 01. 

**Loss function** Edge prediction is essentially geometric inference, rich geometric information will be more helpful. However, the geometric information of an edge includes more than just its endpoints. To improve edge perception, we propose an edge perception enhancement strategy. Specifically, we add a random interpolation point and require the model to predict its interpolation coefficient, enhancing the model’s ability to infer intermediate edge structures. For 

**==> picture [237 x 57] intentionally omitted <==**

**----- Start of picture text -----**<br>
Node Transformer CNN<br>— ——} Plug-and-Play - L e. Encoder<br>Encoder Decoder<br>Constraint a... Plug-and-Play c Constraint e Transformer I<br>Encoder<br>Edge Transformer<br>**----- End of picture text -----**<br>


Figure 3: Constrained generation. Embeddings for constraints are obtained through respective encoders and used as inputs for node generation and edge prediction. Topology constraints use a Transformer-based encoder, while boundary constraints use a CNN-based encoder. 

each edge ( _vi, vj_ ), the interpolation point’s coordinates and attributes are defined as: 

_cλ_ = _λci_ + (1 _− λ_ ) _cj ∈_ [ _−_ 1 _,_ 1)[2] (8) where _rλ_ = **0** _∈_ R[7] , _λ ∼ U_ (0 _,_ 1) is a random interpolation coefficient. The supervised loss for interpolation is: 

**==> picture [169 x 13] intentionally omitted <==**

where _λ_[ˆ] _θ_ represents the model’s predicted interpolation coefficient. _λ_[˜] = 1 _− λ_ , if _λ >_ 0 _._ 5; _λ_[˜] = _λ_ , if _λ ≤_ 0 _._ 5. 

The enhanced edge feature includes the features of the two endpoints of an edge, as well as the random interpolation point. The final edge prediction loss for training is: 

**==> picture [160 x 11] intentionally omitted <==**

where _Lcls_ is the Cross-entropy classification loss. More details are provided in the supplementary material. 

**Floorplan extraction** So far, we have obtained the structural graph _G_ = ( _V, E_ ). We can simply extract all minimal polygonal loops as rooms. Considering that the prediction based on neural networks might contain errors, leading to the absence of a category shared by all nodes, we select the most frequently occurring category as the room type. If multiple categories have the highest frequency, we consider factors such as the rarity of the category and determine the room type based on the following priority: _Storage > Bathroom > Kitchen > Bedroom > Balcony > Living room_ . An illustration is provided in the supplementary material. 

## **Constrained Generation** 

The proposed framework supports constrained floorplan generation. To incorporate constraints into our framework, we introduce constraint encoders to guide the generation. We encode the constraints using an encoder specific to the constraint modality, and the encoded features serve as input to the decoder to model the conditions via cross-attention. To improve the encoding capability of constraint encoders, we train each encoder individually. Specifically, we use a “pretraining + fine-tuning” paradigm, where we first pre-train the constraint encoder on the synthetic constraint data, and then fine-tune the encoder on the real constraint data of the dataset to achieve better generalization. We train constraint encoders in the framework of autoencoder. Without loss of generality, we focus on the boundary-constrained generation and topology-constrained generation in this paper (Figure 3). See more details in the supplementary material. 

**Boundary-constrained generation** A boundary refers to the outer contour of a floorplan, typically represented as a polygon. To encode a boundary, we draw the boundary polygon on an image with a resolution of 256 _×_ 256, converting the boundary polygon into an image. We then use a CNN-based encoder for encoding. Specifically, we modify U-Net (Ronneberger, Fischer, and Brox 2015) by removing skip connections and adding residual connections, as our boundary encoder. The encoder outputs a feature map of 16 _×_ 16 with 1024 channels. During the pre-training phase, we generate random polygons on the image and learn their encodings with a CNN-based autoencoder. In the fine-tuning phase, we train with real boundary data from the dataset. The boundary embeddings are fed to node and edge Transformers, ensuring that the boundary of the generated structural graph adheres to the given constraint of the boundary. 

**Topology-constrained generation** Floorplan topology is used to describe the connectivity between rooms. It is an undirected graph where each node represents a room, and each edge represents a connectivity between two rooms. We encode the topological graph with a Transformer, which outputs 256D embeddings of all rooms, constituting topological embeddings. During the pre-training phase, we randomly generate topological graphs and learn their embeddings using a Transformer-based autoencoder. During the fine-tuning phase, we train with real topological graph data. The topology embeddings are fed to node and edge Transformers, ensuring that the topology of the generated structural graph adheres to the given constraint of the topology. 

## **Experiments** 

Our method is implemented using Pytorch and trained on an NVIDIA GeForce GTX 4090 GPU. To ensure the quality of training at each stage, we train each network separately, using the Adam optimizer (Kingma 2014) with an initial learning rate of 1 _×_ 10 _[−]_[4] . We have used the _RPLAN_ dataset (Wu et al. 2019) for training and testing, which contains more than 80K residential floorplans with dense annotation. The sample size for validing and testing is 3,000 each and the rest is used for training. Creating a vector floorplan takes an average of 0.17 seconds without constraints, 0.67 seconds with boundary constraints, and 0.86 seconds with topological constraints. See more in the supplementary material. 

## **Qualitative Evaluation** 

**Unconstrained generation** Unconstrained generation means that diverse floorplans can be generated without any inputs. The unconstrained generation allows users to explore freely, potentially inspiring more creative and innovative designs. It is worth noting that less research work is currently focused on unconstrained floorplan generation. By tilting balcony walls in the dataset, our method can also generate floorplans with slanted walls. Thanks to our robust structural graph representation, alignment error optimization strategy, and edge perception enhancement strategy, we can generate diverse, high-quality, vector floorplans without any inputs (Figure 4). For more results, please refer to the supplementary materials. 

**Boundary-constrained generation** Both _Graph2Plan_ and _WallPlan_ can generate floorplans from boundaries, thus we compare our method against them. Figure 5 shows a comparison of floorplans generated by different methods. _Graph2Plan_ is prone to generating issues such as unreasonable space divisions and areas, making these defects particularly noticeable as almost every sample exhibits significant flaws. Specifically, the second column features a huge balcony and tiny bedrooms, kitchen, and bathroom; the third column lacks a balcony; the fourth column has a bedroom accessible only through another bedroom on the right, and the fifth column includes an overly narrow balcony. _WallPlan_ , although it produces fewer unreasonable shapes than _Graph2Plan_ , some defects still persist. In the first column, the storage cabinet next to the bathroom should be against a wall, but it is located in the middle of the room; the second column has an unreasonable bathroom division, the third column has overly simplistic divisions, a huge bathroom, and a missing balcony; the fourth column has a super small, impractical bedroom, and the fifth column has a bathroom blocked by storage. The limitations of _Graph2Plan_ and _WallPlan_ are that they can only generate a single result for a specific input, and CNNs struggle to model long-range semantic relationships that involve the reasonableness of room layouts. In contrast, our model, benefiting from structural representation and attention mechanisms, can produce a variety of results that are closer to the fundamental facts of actual buildings. 

**Topology-constrained generation** We compare our method with _House-GAN++_ and _HouseDiffusion_ , which can generate floorplans based on topology graph constraints. It’s noteworthy that _HouseDiffusion_ requires the number of vertices for each room polygon to be pre-specified, hence we retrieve samples from the dataset as its input. Figure 6 shows a comparison of different methods. The principal issue with _House-GAN++_ lies in the peculiar room shapes, and jagged room boundaries are prevalent, leading to weaker visual aesthetics. Specifically, the overly small balcony at the top of the first column, the unreasonable arrangement between balconies and bedrooms in the second, third, and fourth columns, and the bedroom obstructed by the bathroom in the fifth column. And nearly every sample fails to meet the constraints. _HouseDiffusion_ generates better quality compared to _House-GAN++_ , yet it requires the number of vertices for each room polygon to be pre-specified as an extra “constraint”, limiting the diversity of room shapes. Moreover, limited by their binary coordinate optimization, there are issues with boundaries not aligning well, preventing the acquisition of a good wall structure. Many of their generated results fail to satisfy the room’s topological constraints. Specifically, the poorly shaped balconies in the first, third, and fourth columns, the bathroom in the middle of the house in the second column, and the kitchen in the fourth column that can only be entered from a bedroom. In contrast, our method can generate high-quality and diverse room boundaries, making the generated results more natural, aesthetically pleasing, and compliant with constraints. 

**==> picture [439 x 168] intentionally omitted <==**

**----- Start of picture text -----**<br>
Living room Bedroom Kitchen Bathroom Balcony Storage<br>Unconstrained<br>generation<br>Pra yader<br>K P ei ol ie<br>ie oe ee<br>Slanted wall<br>generation<br>**----- End of picture text -----**<br>


Figure 4: A gallery of vector floorplans generated using our framework. From top to bottom: unconstrained generation, topology-constrained generation, boundary-constrained generation, and unconstrained generation with slanted walls. 

**==> picture [507 x 206] intentionally omitted <==**

**----- Start of picture text -----**<br>
Living room Bedroom Kitchen Bathroom Balcony Storage Living room Bedroom Kitchen Bathroom Balcony Storage<br>Ground truth<br>ppaey Buheo House-GAN++<br>Graph2Plan<br>HouseDiffusion<br>WallPlan<br>Ours (27M)<br>Sia & es he Be<br>Ours (106M)<br>Ours<br>fhawnt BEeERGS<br>Figure 6: Comparison with House-GAN++ and HouseDif-<br>Ours on the Our results<br>**----- End of picture text -----**<br>


Figure 6: Comparison with _House-GAN++_ and _HouseDiffusion_ on the topology-constrained generation. Our results exhibit greater diversity, better adherence to constraints, and superior visual quality. 

Figure 5: Comparison with the ground truth, _Graph2Plan_ , and _WallPlan_ on the boundary-constrained generation. Our method can produce more reasonable floorplans. 

mean discrepancy in feature space and is generally considered to be more robust than FID. Table 1 shows the results of the distribution comparison. It’s worth mentioning that we found that the image metrics FID and KID are influenced by a combination of factors such as room type, area, room layout, wall shape, and alignment, which can measure the generation results on the whole, making them excellent indicators for floorplan generation. In the evaluation of boundary-constrained generation, we selected the intersection of the test sets of our model, _Graph2Plan_ , and _WallPlan_ (378 boundaries in total) as input constraints, generating one sample per boundary for each method (as _Graph2Plan_ and _WallPlan_ can only produce a single output). Table 1 shows that the visual, geometric, and other features of our 

## **Quantitative Evaluation** 

**Distribution comparison** The distribution comparison is used to analyze the overall generation capability of a generative model by comparing the differences between the distributions of generated data and real data. We consider the following representative metrics: (1) FID ( _Fr’echet Inception Distance_ )(Heusel et al. 2017) is used to measure the quality and diversity of generated images by calculating the distribution distance between real and generated data in feature space; (2) KID ( _Kernel Inception Distance_ )(Bi´nkowski et al. 2018) uses kernel methods to calculate the maximum 

Table 1: Quantitative evaluation. # Param: parameter counts. Each experiment is repeated 5 times to eliminate randomness, and the average results are reported. The smaller the better for all metrics. The color cyan and magenda mark the top-two results. Void cells: GED is only applicable to topological graph constraints. 

|Constraint<br>Method<br># Param<br>FID<br>KID<br>GED<br>Living<br>Kitchen<br>Bedroom<br>Storage<br>Bathroom<br>Balcony|Constraint<br>Method<br># Param<br>FID<br>KID<br>GED<br>Living<br>Kitchen<br>Bedroom<br>Storage<br>Bathroom<br>Balcony|Constraint<br>Method<br># Param<br>FID<br>KID<br>GED<br>Living<br>Kitchen<br>Bedroom<br>Storage<br>Bathroom<br>Balcony|Constraint<br>Method<br># Param<br>FID<br>KID<br>GED<br>Living<br>Kitchen<br>Bedroom<br>Storage<br>Bathroom<br>Balcony|
|---|---|---|---|
|Topology|_House-GAN++_<br>2M<br>_HouseDiffusion_<br>27M<br>Ours (27M)<br>27M<br>Ours<br>125M|48.40<br>54.66<br>3.9<br>11.87<br>7.23<br>2.59<br>6.64<br>1.62<br>0.57<br>6.82<br>1.79<br>0.49|0.860<br>1.300<br>0.984<br>1.151<br>1.378<br>1.037<br>0.955<br>0.971<br>0.979<br>0.567<br>0.967<br>0.994<br>0.977<br>0.981<br>0.989<br>0.817<br>0.954<br>0.966<br>0.992<br>0.991<br>1.004<br>0.752<br>0.966<br>0.976|
|Boundary|_Graph2Plan_<br>8M<br>_WallPlan_<br>106M<br>Ours (106M)<br>106M<br>Ours<br>137M|8.40<br>1.34<br>-<br>9.07<br>1.02<br>-<br>7.83<br>0.51<br>-<br>7.50<br>0.56<br>-|1.034<br>0.975<br>1.013<br>0.727<br>0.959<br>0.859<br>0.992<br>1.033<br>0.923<br>1.136<br>1.058<br>1.008<br>1.000<br>0.954<br>1.007<br>0.500<br>0.964<br>0.977<br>1.007<br>0.990<br>0.992<br>0.454<br>0.967<br>0.935|



generation results outperform _Graph2Plan_ and _WallPlan_ . For the evaluation of topology graph-constrained generation, we selected the intersection of the test sets of our model, _House-GAN++_ , and _HouseDiffusion_ (a total of 757 topology graphs) as input constraints, generating 757 * 5 samples for each method, calculating the results five times and taking the average. Table 1 indicates that our generation results maintain better geometric consistency, visual appeal, and better practicality. Additionally, for topology-constrained generation, we reference _HouseDiffusion_ (Shabani, Hosseini, and Furukawa 2023), and introduce Graph Edit Distance (GED) (Abu-Aisheh et al. 2015) as an additional metric for evaluation. GED is a graph-matching approach that calculates the distance between the input bubble diagram and the one reconstructed from the generated floorplan. For GED of _House-GAN++_ , we directly use the reported GED in their paper. Table 1 indicates that our generation results maintain better constraint satisfaction, which benefits from our structural representation. For the unconstrained generation of floorplans with slanted walls, FID=12.02, KID=9.98. 

**Statistics comparison** We also conducted a statistical analysis of the generated vector results to evaluate the quality of the generated floorplans in terms of practicality. We ran the test set five times for each method. For each method’s generated results, we calculated the amount of each type of room. We calculated the average values of these statistics and compared them with the corresponding statistics of the ground truth in the real dataset (the closer to 1, the better). The results, as shown in Table 1, indicate that our method has a clear advantage in practicality compared to state-ofthe-art techniques. For most types of rooms (living room, kitchen, bedroom), the amount generated by our method is closer to the real dataset. Only the balcony, storage, and bathroom are closer to the dataset by _House-GAN++_ and _HouseDiffusion_ , but the difference in closeness with ours is not significant. Moreover, for boundary constraints, our method also shows an advantage in bathrooms. 

## **Ablation Study** 

We introduce two geometric enhancement strategies: one for alignment enhancement that optimizes the alignment error 

Table 2: Ablation Study. FE (Fake Edge) is the ratio of misclassified edges, and AE (Alignment Error) is the node alignment error. Each experiment is repeated 5 times to eliminate randomness, and the average results are reported. The smaller the better for all metrics. The best results are highlighted in **bold** . 

|Method|FID|KID|FE(%)|AE|
|---|---|---|---|---|
|**Node**_None_ **+ Edge**_None_|6.23|4.12|1.35|0.34|
|**Node**_Real_ **+ Edge**_None_|6.18|4.16|1.32|0.29|
|**Node**_Binary_ **+ Edge**_None_|5.90|4.03|1.28|0.33|
|**Node + Edge**_None_|**4.96**|**2.92**|**1.15**|**0.23**|
|**Node**_None_ **+ Edge**|5.72|3.74|1.13|0.33|
|**Node**_Real_ **+ Edge**|5.84|3.93|1.11|0.29|
|**Node**_Binary_ **+ Edge**|5.84|3.96|1.13|0.34|
|**Node + Edge**(Ours)|**4.83**|**2.84**|**0.95**|**0.23**|



of nodes in mixed-base representations, and the other for perception enhancement that enhances the geometric perception ability of our edge prediction. To evaluate these two strategies, we have conducted a series of ablation experiments (Table 2). In these experiments, **Node** _None_ indicates no alignment enhancement for node generation, **Node** _Real_ indicates the only use of continuous real-valued alignment error optimization, **Node** _Binary_ indicates the only use of binary discrete alignment error optimization, **Edge** _None_ indicates no perception enhancement for edge prediction, and **Node** and **Edge** is our full methods. 

Our method ( **Node + Edge** ) outperforms the other ablation methods in terms of all evaluated metrics, demonstrating that our geometric enhancements effectively improve the quality of the generated floorplans. For FID and KID, our method achieved the lowest FID=4.83 and KID=2.84, indicating that the generated floorplans are more similar to the real data distribution. This demonstrates that both geometric alignment and edge perception enhancements contribute to higher visual fidelity and distribution alignment. FE is minimized with our method (0.95%), which means the struc- 

**==> picture [235 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
Living room Bedroom Kitchen Bathroom Balcony Storage<br>Generated<br>Qease<br>Euclidean<br>Wasserstein<br>**----- End of picture text -----**<br>


Table 3: Quantitative evaluation on the _LIFULL_ dataset. 

|Dataset|FID|KID|FE(%)|FE(%)|AE|
|---|---|---|---|---|---|
|_LIFULL_|12.44|3.61||6.11|3.75|
|Living room<br>Bedroom|Kitchen|Bathroom||Balcony|Closet|
|Restroom<br>Corridor|Washing room|Washing room<br>Pipe space||Unknown||
|LIFULL<br>BEag|BEag|BEag||t=|t=|



Figure 7: Retrieval Analysis. Generated: the generated floorplans. Euclidean(Wasserstein): the closest training floorplans based on Euclidean(Wasserstein) distance. 

Figure 8: Qualitative evaluation on the _LIFULL_ dataset. 

tural integrity and accuracy of the generated floorplans are effectively improved. AE remains consistently low (0.23), highlighting that our method effectively maintains geometric consistency and precision in the generated floorplans. 

## **Retrieval Analysis** 

To further evaluate the generalization ability of the model, we performed a comprehensive retrieval analysis of the generated floorplans against those in the training set. For each generated floorplan (3,000 in total), we compute the minimum Euclidean distance and Wasserstein distance in image space between it and all floorplans in the training set: if the training set sample corresponding to the minimum distance is visually similar to the generated floorplan, it indicates possible overfitting. The retrieval results show that there is a significant difference between the results we generate and those retrieved from the dataset, i.e., our model has good generalization performance. However, the distance values are not intuitive enough. To illustrate the difference intuitively, we provide several examples in Fig. 7 showing the closest match between the generated floorplans and the ones in the training set. 

## **Discussion on Model Sizes** 

To ensure a fair comparison, we evaluated our method under equivalent model parameter sizes. 

**Topology-constrained generation** Our original model for topology-constrained generation has approximately 125 million parameters, while _House-GAN++_ and _HouseDiffusion_ have about 2 million and 27 million parameters, respectively. To assess the impact of model size, we reduced our model’s parameter count to 27 million to match that of _HouseDiffusion_ , retraining it with all other configurations unchanged. As shown in Figure 6 and Table 1, even with the same parameter size, our method still achieves superior performance. This indicates that the effectiveness of our method stems from its inherent advantages rather than a larger parameter count. Regarding _House-GAN++_ , we could not adjust our model to match their 2 million parameters. However, 

_House-GAN++_ has inherent problems. See the supplementary material for details. 

**Boundary Constrained Generation** Our model has about 137 million parameters. In comparison, _WallPlan_ has about 106 million parameters and _Graph2Plan_ has about 8 million parameters. To evaluate the impact of model size, we reduce the number of parameters of our model to 106 million to match _WallPlan_ and retrain it. As shown in Fig. 5 and Table 1, our method still achieves superior performance, which also shows that our method has its inherent advantages. Regarding _Graph2Plan_ , we were unable to adjust the model to match their 8 million parameters. However, _Graph2Plan_ has inherent problems. See the supplementary material for details. 

## **Evaluation on Other Datasets** 

To further demonstrate the generalization and effectiveness of our model, we conducted experiments on the _LIFULL_ dataset (LIFULL Co. 2016). We obtained the vectorized subset of _LIFULL_ dataset via _Raster-to-Graph_ (Hu et al. 2024), which contains 10,804 floorplans: 500 are used for validation, 500 for testing, and the rest for training. We trained our unconstrained model using the same configurations as with _RPLAN_ , adjusting only the room categories to align with the 12 in _LIFULL_ . Table 3 and Fig. 8 showcases the quantitative and qualitative results, respectively, demonstrating our method’s generalizability and effectiveness in diverse scenarios. However, as the _Raster-to-Graph_ authors themselves mentioned, the annotation quality of the _LIFULL_ vectorized dataset they extracted is weaker and the number is much smaller than that of _RPLAN_ , which make the effect on _LIFULL_ limited. Possible future works would include extending our method to the _MSD_ (van Engelenburg et al. 2024) dataset, which contains a significant share of layouts of multi-apartment dwellings. _MSD_ lacks vectorized structural graphs, and the presence of complex shapes of walls in the images makes vectorized structural graphs difficult to extract. 

## **Conclusion** 

We introduce a novel vector floorplan generation framework, _GSDiff_ , which converts the complex floorplan generation problem into a structural graph generation problem, further decoupled into node generation and edge prediction. Additionally, we incorporate geometric enhancements into the generation framework. By optimizing the node alignment error, we achieve better geometric consistency. Through edge perception enhancement, we improve the edge prediction ability, resulting in better geometric plausibility. Experiments have shown that _GSDiff_ outperforms existing state-of-the-art methods. 

However, _GSDiff_ faces limitations, such as reliance on quality training data for learning complex node and edge dependencies and scalability challenges in larger projects. Future research directions include exploring semi-supervised learning to reduce data dependency, enhancing constraint handling for complex scenarios and improving scalability. These aim to broaden the framework’s capabilities and applicability in architectural design. 

## **Acknowledgments** 

We would like to thank the anonymous reviewers for their constructive suggestions and comments. This work is supported by the National Natural Science Foundation of China (62102126, 62372153) and the Fundamental Research Funds for the Central Universities of China (JZ2023HGTB0269, JZ2022HGQA0163). In this paper, we used ”LIFULL HOME’S Dataset” provided by LIFULL Co., Ltd. via IDR Dataset Service of National Institute of Informatics. 

## **References** 

Abu-Aisheh, Z.; Raveaux, R.; Ramel, J.-Y.; and Martineau, P. 2015. An exact graph edit distance algorithm for solving pattern recognition problems. In _4th International Conference on Pattern Recognition Applications and Methods 2015_ . 

Bi´nkowski, M.; Sutherland, D. J.; Arbel, M.; and Gretton, A. 2018. Demystifying mmd gans. _arXiv preprint arXiv:1801.01401_ . 

Bisht, S.; Shekhawat, K.; Upasani, N.; Jain, R. N.; Tiwaskar, R. J.; and Hebbar, C. 2022. Transforming an adjacency graph into dimensioned floorplan layouts. In _Computer Graphics Forum_ , volume 41, 5–22. Wiley Online Library. 

Chaillou, S. 2020. Archigan: Artificial intelligence x architecture. In _Architectural intelligence: Selected papers from the 1st international conference on computational design and robotic fabrication (CDRF 2019)_ , 117–127. Springer. 

Chen, J.; Zhang, R.; Zhou, Y.; and Chen, C. 2024. Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints. _arXiv preprint arXiv:2402.04754_ . 

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in neural information processing systems_ , 30. 

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33: 6840–6851. 

Ho, J.; Saharia, C.; Chan, W.; Fleet, D. J.; Norouzi, M.; and Salimans, T. 2022. Cascaded diffusion models for high fidelity image generation. _Journal of Machine Learning Research_ , 23(47): 1–33. 

Hu, R.; Huang, Z.; Tang, Y.; Van Kaick, O.; Zhang, H.; and Huang, H. 2020. Graph2plan: Learning floorplan generation from layout graphs. _ACM Transactions on Graphics (TOG)_ , 39(4): 118–1. 

Hu, S.; Wu, W.; Su, R.; Hou, W.; Zheng, L.; and Xu, B. 2024. Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer. _Computer Graphics Forum_ , 43(2): e15007. 

Kingma, D. 2014. Adam: a method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ . 

Laignel, G.; Pozin, N.; Geffrier, X.; Delevaux, L.; Brun, F.; and Dolla, B. 2021. Floor plan generation through a mixed constraint programming-genetic optimization approach. _Automation in Construction_ , 123: 103491. 

LIFULL Co., L. . 2016. LIFULL HOME’S High Resolution Floor Plan Image Data. Informatics Research Data Repository, National Institute of Informatics. (dataset). https: //doi.org/10.32130/idr.6.2. 

Liu, H.; Yang, Y.-L.; AlHalawani, S.; and Mitra, N. J. 2013. Constraint-aware interior layout exploration for precast concrete-based buildings. _The Visual Computer_ , 29: 663–673. 

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 9298–9309. 

Merrell, P.; Schkufza, E.; and Koltun, V. 2010. Computergenerated residential building layouts. In _ACM SIGGRAPH Asia 2010 papers_ , 1–12. 

Nauata, N.; Chang, K.-H.; Cheng, C.-Y.; Mori, G.; and Furukawa, Y. 2020. House-gan: Relational generative adversarial networks for graph-constrained house layout generation. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16_ , 162–177. Springer. 

Nauata, N.; Hosseini, S.; Chang, K.-H.; Chu, H.; Cheng, C.Y.; and Furukawa, Y. 2021. House-gan++: Generative adversarial layout refinement network towards intelligent computational agent for professional architects. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 13632–13641. 

Nichol, A.; Dhariwal, P.; Ramesh, A.; Shyam, P.; Mishkin, P.; McGrew, B.; Sutskever, I.; and Chen, M. 2021. Glide: Towards photorealistic image generation and editing with textguided diffusion models. _arXiv preprint arXiv:2112.10741_ . 

Nichol, A.; Jun, H.; Dhariwal, P.; Mishkin, P.; and Chen, M. 2022. Point-e: A system for generating 3d point clouds from complex prompts. _arXiv preprint arXiv:2212.08751_ . 

Para, W.; Guerrero, P.; Kelly, T.; Guibas, L. J.; and Wonka, P. 2021. Generative layout modeling using constraint graphs. In _Proceedings of the IEEE/CVF international conference on computer vision_ , 6690–6700. 

Poole, B.; Jain, A.; Barron, J. T.; and Mildenhall, B. 2022. Dreamfusion: Text-to-3d using 2d diffusion. _arXiv preprint arXiv:2209.14988_ . 

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 10684– 10695. 

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U- Net: Convolutional Networks for Biomedical Image Segmentation. In Navab, N.; Hornegger, J.; Wells, W. M.; and Frangi, A. F., eds., _Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015_ , 234–241. Cham: Springer International Publishing. ISBN 978-3-31924574-4. 

Saharia, C.; Chan, W.; Chang, H.; Lee, C.; Ho, J.; Salimans, T.; Fleet, D.; and Norouzi, M. 2022. Palette: Image-to-image diffusion models. In _ACM SIGGRAPH 2022 conference proceedings_ , 1–10. 

Shabani, M. A.; Hosseini, S.; and Furukawa, Y. 2023. Housediffusion: Vector floorplan generation via a diffusion model with discrete and continuous denoising. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 5466–5475. 

Shekhawat, K.; Upasani, N.; Bisht, S.; and Jain, R. N. 2021. A tool for computer-generated dimensioned floorplans based on given adjacencies. _Automation in Construction_ , 127: 103718. 

Sun, J.; Wu, W.; Liu, L.; Min, W.; Zhang, G.; and Zheng, L. 2022. Wallplan: synthesizing floorplans by learning to generate wall graphs. _ACM Transactions on Graphics (TOG)_ , 41(4): 1–14. 

van Engelenburg, C.; Mostafavi, F.; Kuhn, E.; Jeon, Y.; Franzen, M.; Standfest, M.; van Gemert, J.; and Khademi, S. 2024. MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes. arXiv:2407.10121. 

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L. u.; and Polosukhin, I. 2017. Attention is All you Need. In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.; Vishwanathan, S.; and Garnett, R., eds., _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc. 

Wang, X.-Y.; and Zhang, K. 2020. Generating layout designs from high-level specifications. _Automation in Construction_ , 119: 103288. 

Wu, W.; Fan, L.; Liu, L.; and Wonka, P. 2018. Miqp-based layout design for building interiors. In _Computer Graphics Forum_ , volume 37, 511–521. Wiley Online Library. 

Wu, W.; Fu, X.-M.; Tang, R.; Wang, Y.; Qi, Y.-H.; and Liu, L. 2019. Data-driven interior plan generation for residential buildings. _ACM Transactions on Graphics (TOG)_ , 38(6): 1–12. 

## **Reproducibility Checklist** 

This paper: 

Includes a conceptual outline and/or pseudocode description of AI methods introduced (yes) 

Clearly delineates statements that are opinions, hypothesis, and speculation from objective facts and results (yes) 

Provides well marked pedagogical references for lessfamiliare readers to gain background necessary to replicate the paper (yes) 

Does this paper make theoretical contributions? (no) 

Does this paper rely on one or more datasets? (yes) 

If yes, please complete the list below. 

A motivation is given for why the experiments are conducted on the selected datasets (yes) 

All novel datasets introduced in this paper are included in a data appendix. (NA) 

All novel datasets introduced in this paper will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. (NA) 

All datasets drawn from the existing literature (potentially including authors’ own previously published work) are accompanied by appropriate citations. (yes) 

All datasets drawn from the existing literature (potentially including authors’ own previously published work) are publicly available. (yes) 

All datasets that are not publicly available are described in detail, with explanation why publicly available alternatives are not scientifically satisficing. (NA) 

Does this paper include computational experiments? (yes) If yes, please complete the list below. 

Any code required for pre-processing data is included in the appendix. (no). 

All source code required for conducting and analyzing the experiments is included in a code appendix. (no) 

All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. (yes) 

All source code implementing new methods have comments detailing the implementation, with references to the paper where each step comes from (partial) 

If an algorithm depends on randomness, then the method used for setting seeds is described in a way sufficient to allow replication of results. (NA) 

This paper specifies the computing infrastructure used for running experiments (hardware and software), including GPU/CPU models; amount of memory; operating system; names and versions of relevant software libraries and frameworks. (yes) 

This paper formally describes evaluation metrics used and explains the motivation for choosing these metrics. (yes) 

This paper states the number of algorithm runs used to compute each reported result. (yes) 

Analysis of experiments goes beyond single-dimensional summaries of performance (e.g., average; median) to include measures of variation, confidence, or other distributional information. (yes) 

The significance of any improvement or decrease in performance is judged using appropriate statistical tests (e.g., Wilcoxon signed-rank). (no) 

This paper lists all final (hyper-)parameters used for each model/algorithm in the paper’s experiments. (yes) 

This paper states the number and range of values tried per (hyper-) parameter during development of the paper, along with the criterion used for selecting the final parameter setting. (yes) 

# **GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation - Supplements** 

**Sizhe Hu**[1*] **, Wenming Wu**[2†‡] **, Yuntao Wang**[3§] **, Benzhu Xu**[4¶] **, Liping Zheng**[5||] 

1Hefei University of Technology 485 Danxia Lu, Shushan District, Hefei City, Anhui Province, 230061, China 

## **Preliminary** 

Diffusion models are a class of generative models that train neural networks to cleverly reverse a noise-adding process, enabling them to generate samples that simulate the original samples of the dataset from simple noise. A standard diffusion model typically consists of a forward process and a reverse process. 

In the forward process, noise is gradually added to the original sample _V_ 0 over _T_ steps, resulting in a noisy sample _VT_ that resembles Gaussian noise. For a given time step _t_ , this process generates a noisier version _Vt_ using the following equation: 

**==> picture [172 x 11] intentionally omitted <==**

**==> picture [162 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
FFN<br>×  L Add<br>FFN<br>Norm<br>Attention<br>Add<br>Global<br>Self-Attention<br>=<br>Norm<br>Attention Mask<br>Node Embeddings<br>**----- End of picture text -----**<br>


where ~~_α_~~ _t_ = _α_ 1 _α_ 2 _· · · αt_ is the cumulative product of coefficients _αi_ used to control the proportion of the signal _V_ 0 in _Vt_ , and _ϵ_ is standard Gaussian noise. As _t_ increases, the proportion of noise becomes larger until it approaches the pure Gaussian noise. 

The reverse process starts with Gaussian noise and gradually denoises it to reconstruct the original sample _V_ 0. Less noisy versions of the original sample are recovered step by step using a neural network parameterized by _θ_ , which outputs _ϵθ_ as an estimation of the added noise. The probability distribution at time _t −_ 1 for the noisy sample _Vt_ is given by 

**==> picture [200 x 12] intentionally omitted <==**

where 

Figure 1: The network architecture of the node Transformer. 

## **Node generation** 

## **Network architecture** 

The network architecture of the node Transformer (Figure 1) consists of stacked Transformer decoder layers (× _L_ ). The input to the node Transformer is the node set _Vt_ and the time step _t_ . For a node _vi_ = ( _ci, ri, bi_ ) _∈_ [ _−_ 1 _,_ 1)[2+7+1] , we first obtain the node embedding _fi ∈_ R _[d]_ , where _d_ denotes the embedding dimension. Specifically, for the positional coordinates of the node, we first denormalize _ci_ = ( _xi, yi_ ) _∈_ [ _−_ 1 _,_ 1)[2] back to [0 _,_ 255][2] and use the positional encoding proposed by (Vaswani et al. 2017), _fi[c]_[=][[] _[γ]_[(] _[x][i]_[)] _[, γ]_[(] _[y][i]_[)]] _[∈]_ R _[d]_ , where _γ_ ( _t_ ) is defined as 

**==> picture [230 x 11] intentionally omitted <==**

- *https://orcid.org/0000-0001-9308-5768 

**==> picture [210 x 25] intentionally omitted <==**

- †Corresponding author: wwming@hfut.edu.cn (Wenming Wu) ‡https://orcid.org/0000-0002-0640-8520 

- §https://orcid.org/0009-0003-2827-5727 

- ¶https://orcid.org/0000-0001-7756-0901 

_σ_[2] is the variance associated with the diffusion process and _N_ denotes the Gaussian distribution. 

- ||https://orcid.org/0000-0001-5071-9628 

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

4 _j_ where _k_ = _d_ 4 _[−]_[1][,] _[ω][j]_[=] 100001 _d_ , and _d_ is the embedding dimension. For the se antic d background attributes ( _ri, bi_ ) of the node, we use a fully connected layer to map ( _ri, bi_ ) into a _d_ -dimensional space to obtain the embedding _fi_[(] _[r,b]_[)] _∈_ R _[d]_ . The time step _t_ is encoded to represent the noise level, and we use FFN to obtain the time embedding _fi[t][∈]_[R] _[d]_[.][The][final][node][embedding][of] _[v][i]_[is][obtained][by] fusing all three embeddings 

**==> picture [164 x 14] intentionally omitted <==**

The network architecture of the node Transformer consists of multiple decoder layers, with the input being the node embeddings of all nodes. Each decoder layer includes normalization, multi-head global self-attention, residual connections, and FFN, where the attention module models the dependencies between all nodes. Considering the variable number of nodes of each sample, we use background nodes to pad the node set to the same shape. The output of the decoder is a set of node embeddings, which, after passing through several decoder layers, are used to predict the noise parameters _ϵθ_ of _pθ_ ( _Vt−_ 1 _|Vt_ ) at the next time step. 

## **Alignment loss** 

High-quality structural graphs of architectural floorplans require geometric consistency, meaning nodes must align well. The alignment loss is used to measure the alignment error between nodes of the structural graph. Inspired by (Li et al. 2020), we define the alignment error of each node as the minimum distance between that node and all other nodes in any direction. Optimizing node alignment can be achieved by optimizing the sum of alignment errors of all nodes. We take the sum of alignment errors of all nodes as the alignment loss. 

We use the predicted noise _ϵθ_ and Equation 1 to predict node set _V_[ˆ] 0. For _V_[ˆ] 0, we have 

**==> picture [194 x 28] intentionally omitted <==**

where _n_ denotes the number of nodes in the node set, _g_ ( _x_ ) = _−d·_ log(1 _−[x] d_[)][,][ ∆] _[b] i[∗]_[= min] _[∀][j]_[=] _[i][ |][b] i[∗][−][b][∗] j[|]_[,] _[ ∗∈A]_[ =] _[ {][X, Y][ }]_[,] and _d_ represents the maximum allowable distance in direction _∗_ . Since the range of the positional coordinate for each node is [ _−_ 1 _,_ 1), _d_ = 1 _−_ ( _−_ 1) = 2. 

However, the inherent regression errors of neural networks present a challenge. Directly optimizing the regression loss, typically the MSE loss) of the neural network does not work well. One solution is to use a discrete coordinate representation to optimize the alignment loss. _HouseDiffusion_ (Shabani, Hosseini, and Furukawa 2023) employs an 8- bit binary integer representation for coordinates within the range [0 _,_ 255], which, in theory, could facilitate the model in learning precise alignment. However, this approach introduces notable risks in practice. For instance, consider a coordinate represented by the binary value [1 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _,_ 1 _,_ 1][2] (131). If a prediction error occurs at the 4[th] bit, the actual predicted result would be [1 _,_ 0 _,_ 0 _,_ 0 _,_ 1 _,_ 0 _,_ 1 _,_ 1][2] (139), causing the coordinate to erroneously jump from 131 to 139. This 

**==> picture [131 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
min(ΔbiX, ΔbiY) = 8.39 Real-value<br>Base [4] (·)<br>|<br>a: (0020.120331130...) : 4<br>[0,255]<br>| “en  ε<0.1<br>0 + 0 + 2 + 0 + 1 + 2 = 5 *<br>g [4] (x) = -18 log(1 - x/18)<br>| ig i<br>5.86<br>qH (3333.33)4 > 3*6=18 [:]<br>**----- End of picture text -----**<br>


Figure 2: Quaternary alignment error. **Left** : The calculation process of a quaternary alignment error. **Right** : Plot of _g_[4] ( _x_ ) = _−_ 18 log 1 _−_ 18 _[x]_[.] _[g]_[4][(] _[x]_[)][is][used][to impose][penal-] ties on larger alig ent e ors. When the independent variable (the alignment error of a single node) approaches 0, _g_[4] ( _x_ ) approximates _x_ ; when the independent variable approaches _x_ = _d_[4] = 18, an infinite penalty will be imposed. The larger the independent variable, the greater the penalty. 

magnitude of error significantly exceeds what would typically occur with direct regression methods. 

To harmonize the advantages of binary representation and alignment loss, we convert the alignment error of each node into the binary form for regression. The discreteness introduced by the binary form helps to suppress noise to some extent, while the alignment loss constrains larger deviations. This method effectively avoids the pitfalls of high-bit inaccuracies and promotes coordinate alignment through discretization. The binary alignment loss is as follows 

**==> picture [233 x 40] intentionally omitted <==**

where _Base_[2] ( _·_ ) is the binary representation, ∆ _b[∗] i_ = min _j_ = _i |b[∗] i[−][b][∗] j[|]_[,] _[ ∗∈{][X, Y][ }]_[,] _[ g]_[2][(] _[x]_[)][=] _[−][d]_[2][ log][1] _[ −] d[x]_[2][,] with _d_[2] indicating the maximum allowable distance under the binary representation. _n_ is the node number, _s_ is the bit size. To calculate the binary alignment loss, we convert the coordinate range from [ _−_ 1 _,_ 1) to [0 _,_ 255] (corresponding to 8 bits). We take a precision of 1 _×_ 10 _[−]_[1] (corresponding to 4- bit binary fraction), so _s_ is set to 12 in our experiments. We use the L1 norm of the binary vector to represent the binary distance, therefore, 

**==> picture [250 x 52] intentionally omitted <==**

As we convert the coordinate range from [ _−_ 1 _,_ 1) to [0 _,_ 255], we finally normalize the binary alignment loss by a scaling factor of 1 _/_ 128. The notation ( _·_ ) _j_ represents the _j_ -th bit. 

Binary learning treats high and low bits equally important, which is not optimal for gradient-based optimization. To balance precision and optimizability, we propose a mixed-base optimization strategy, combining multiple numeric bases: we trade off computational complexity (which increases linearly with the number of bases) and representation granu- 

larity by adding quaternary, octal, and hexadecimal bases to the binary and real number representations. The quaternary, octal, and hexadecimal losses are similar to those in binary. The bit size _s_ is 6, 5, and 3 for these three bases, respectively. The L1 norm of the corresponding base vectors _d[k]_ is 18, 35, and 45, respectively. The normalization scaling factor is also 1 _/_ 128. Without loss of generality, we illustrate the calculation process of a quaternary alignment error in Figure 2. 

The final mixed-base alignment loss is given by 

**==> picture [215 x 24] intentionally omitted <==**

_LACE_ (Chen et al. 2024) suggests applying alignment loss at larger time _t_ , which may degrade performance, so they use a time-dependent constraint weight, applying larger alignment loss weights only at smaller time _t_ . We also apply this weight function, multiplying it with the alignment loss: _ω_ ( _t_ ) = 1 _− αT −t_ where _αT −t_ is from Equation 1. 

## **Clamping** 

During sampling (i.e., the reverse process of the diffusion model), the neural network’s estimation of data at _t_ = 0 may exceed reasonable ranges, causing deviation in the sampling path and degrading generation quality. The clamping technique limits the neural network’s output to reasonable ranges to prevent such occurrences. In our case, the normalized coordinates for node attributes are limited to [ _−_ 1 _,_ 1)[2] , and other attributes are constrained to _{_ 0 _,_ 1 _}_ . According to Equation 2, the predicted noise _ϵθ_ in the reverse process will be converted into the prediction _V_[ˆ] 0. We clamp _V_[ˆ] 0 within these ranges at each step of the reverse process to improve generation quality. We set the threshold of all semantic attributes to 0.5 and the threshold of background attributes to 0.75 to avoid missing nodes. 

## **Edge prediction** 

## **Network architecture** 

We use the edge Transformer to perform binary classification (“True/False”) on the candidate edges. The network architecture of the edge Transformer (Figure 3) consists of stacked Transformer decoder layers (× _L_ ), each comprising normalization, multi-head global self-attention, residual connections, and FFN. The input to the Edge Transformer consists of the embeddings of the candidate edge set _E[′]_ and the padding mask constructed for _E[′]_ , where an edge is considered non-padding only if both of its endpoints are nonpadding. After passing through several decoder layers, the output embeddings of the candidate edges are fed through an FFN to predict the authenticity of the edges. All edges identified as “True” form the edge set _E_ , which is the output of the edge prediction. 

## **Edge perception enhancement** 

In general, the edge prediction model only uses the features of two endpoints as the edge features, which lack sufficient geometric information, resulting in missing or false edges and making the edge prediction unreasonable. To enhance 

**==> picture [173 x 207] intentionally omitted <==**

**----- Start of picture text -----**<br>
＾<br>True/False λθ<br>FFN FFN<br>×  L Add<br>FFN<br>Norm<br>Attention<br>Add<br>Global<br>Self-Attention<br>Norm<br>Attention Mask<br>Padding<br>Edge Embeddings<br>**----- End of picture text -----**<br>


Figure 3: The network architecture of the edge Transformer. 

the ability of edge perception, we propose a self-supervised edge perception enhancement strategy, as shown in Figure 4. Specifically, for each candidate edge, we introduce a third point at a random location on the edge, determined by a random coefficient _λ_ . As a result, each edge is no longer represented solely by two endpoints, but by two endpoints along with a randomly interpolated point. The model is trained to predict this interpolation coefficient _λ_ , which forces the model to better perceive edges. This strategy improves the geometric reasoning capability of the model by introducing additional geometric features for each edge. 

For each candidate edge ( _vi, vj_ ), the enhanced edge features include the features of both endpoints, as well as the feature combination of the random interpolation point. The interpolation point feature is determined by the interpolation coefficient _λ_ , which decides the location of the interpolation point on the edge. The coordinates and semantics of the interpolation point are as follows 

**==> picture [187 x 12] intentionally omitted <==**

**==> picture [147 x 12] intentionally omitted <==**

where _λ ∼ U_ (0 _,_ 1) is a uniformly distributed interpolation coefficient sampled from the range [0 _,_ 1]. The zero vector **0** represents semantics, as the semantics at the random interpolation point on any candidate edge do not contribute to geometric reasoning. The loss for the self-supervised term is defined as 

**==> picture [174 x 25] intentionally omitted <==**

**==> picture [169 x 19] intentionally omitted <==**

where _λ_[ˆ] _θ_ ( _eij_ ) represents the predicted interpolation coefficient. We process the original interpolation coefficient _λ ∈_ [0 _,_ 1] as follows: if _λ >_ 0 _._ 5, we use 1 _− λ_ ; otherwise, we 

**==> picture [165 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.24a ~<br>a λ=0.24<br>0.76a<br>b<br>0.54b 0.46b ~λ=0.46<br>-<br>{<br>}<br>**----- End of picture text -----**<br>


Figure 4: Edge perception enhancement. This figure shows the relationship between _λ_[˜] and the position of the interpolation point for candidate edges of lengths a and b. For the candidate edge of length a, the interpolation point divides the candidate edge into two segments with a length ratio of 0.76:0.24 (regardless of whether the interpolation coefficient _λ_ is 0.24 or 0.76), then _λ_[˜] is equal to the smaller of the two, which is 0.24. The interpolation coefficient _λ_ is random each time, ensuring that every point on the entire candidate edge can be selected. 

**==> picture [208 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
[      ] [          ]<br>[      ] [  ]<br>Living room Bathroom Kitchen<br>oe Bedroom Balcony Storage<br>**----- End of picture text -----**<br>


Figure 5: An example of floorplan extraction. 

keep it as is. This is done because the edges are undirected, so the model cannot distinguish which endpoint is the reference point during prediction. The total loss is defined as 

**==> picture [157 x 11] intentionally omitted <==**

where _Lcls_ is the Cross-entropy classification loss, and _Lλ_ is the interpolation coefficient regression loss. 

It is noteworthy that our edge perception enhancement strategy differs significantly from the mixup technique (Zhang et al. 2017) in both implementation and objective. The mixup technique creates new samples by blending two random samples from the training data, aiming to improve the model’s generalization and its resistance to adversarial samples. In contrast, our strategy is tailored to enhance the model’s perception of the geometric characteristics of edges within a structure. We assign a distinct interpolation coefficient to each candidate edge and train the model to predict this coefficient, enabling the model to discern the continuity of points along the edge, treating the candidate edge as a whole line segment rather than just two endpoints. This is crucial for the edge prediction task, which involves geometric inference. Unlike the mixup technique, whose goal is to smooth the decision boundary in the model’s output space, our strategy focuses on enriching the input feature space with geometric features that are directly pertinent to the current task. 

## **Floorplan extraction** 

The floorplan extraction process is easy to implement. We can simply extract all minimal polygonal cycles as rooms. This process involves starting from a certain edge in the structural graph, following a predefined order of nodes, moving from the smaller-numbered endpoint to the largernumbered endpoint, and consistently turning the maximum 

angle in a fixed clockwise (or counterclockwise) direction to select the corresponding next edge. This process continues until it returns to the starting edge, and all edges traversed during this process form a minimal polygonal cycle. This procedure is illustrated in Figure 5. Specifically, we traverse all edges to obtain all polygonal cycles and select the nonduplicate ones. In practice, we apply the following simple rule to avoid obtaining duplicate polygons: for each edge, if it has been traversed in the order from the smaller-numbered endpoint to the larger-numbered endpoint, it is marked as visited and will not be visited again. Each time, we only traverse edges that have not been visited. 

## **Constrained generation** 

## **Training paradigm** 

We do not directly train the constraint encoder and node generation model jointly, as this would limit training on a training dataset of finite size. Given the critical importance of the quality of the constraint encoder, we first train each constraint encoder separately to achieve near-lossless performance, and then train the constraint encoder and node generation model jointly. To train the constraint encoder, we adopt a pre-training + fine-tuning paradigm: we first perform pretraining using randomly generated samples (which can be considered as having an infinite size of the training dataset), and then fine-tune the model on our training dataset. This strategy enhances generalization capability. The training is based on the reconstruction loss of the autoencoder. 

## **Boundary-constrained generation** 

In architectural floorplans, the boundary refers to the contour formed by the outer walls of the building, typically rep- 

**==> picture [472 x 219] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 64 64 Encoder Room Class Adjacent Relation<br>64 128128<br>128 256 256 256 512 512 512 1024 1024<br>FFN FFN<br>i L1 Loss ter 128 256 512 1024Decoder poolingconvolutionupsampleidentity O ×  M o<br>FFN<br>Attention<br>256 256 256 512 512 512 512 1024 Masked Norm<br>128128128 256<br>Gift |<br>3 64 64 64 128<br>Topological<br>Figure 6: The network architecture of the autoencoder for Self-Attention<br>boundary constraints. Attention Mask Norm<br>resented as a polygon. Given its geometric nature, we use the —<br>Convolutional Neural Network (CNN) to encode it. Specifi-<br>cally, during the pre-training phase, the input to the CNN is<br>G m<br>heuristically constructed. We determine each polygon vertex<br>Topological Graph Room Embeddings<br>**----- End of picture text -----**<br>


Figure 6: The network architecture of the autoencoder for boundary constraints. 

resented as a polygon. Given its geometric nature, we use the Convolutional Neural Network (CNN) to encode it. Specifically, during the pre-training phase, the input to the CNN is heuristically constructed. We determine each polygon vertex through a random walk on a 256 _×_ 256 three-channel blank image, where each vertex’s 2D coordinates are uniformly sampled across the entire image. The edges of the polygon are drawn in layers with different colors: green (7 pixels), blue (5 pixels), red (3 pixels), and black (1 pixel). The number of vertices, corresponding to the number of random walk steps, is sampled from the training dataset. 

Figure 7: The network architecture of the topology Transformer. Gray in attention mask: which attention are masked. 

Transformer is _G_ top and the cross-attention mask, where the room set _V_ top attends to the given adjacency relationships _E_ top. After passing through several encoder layers, the output is the room embeddings. 

We modified the U-Net structure (Ronneberger, Fischer, and Brox 2015) by removing the skip connections and converting it into an autoencoder, adding residual connections between layers to improve performance. As the network depth increases, the number of channels in the encoder gradually increases, and we replace identity mapping with 1 _×_ 1 convolution. The network output is also a 256 _×_ 256 threechannel image, and we compute the L1 loss between the input and output images. Figure 6 illustrates the network architecture of the autoencoder for boundary constraints. For fine-tuning, we use the real boundary sample from the training dataset. The boundaries are drawn with black lines 7 pixels wide, and the same loss is used for training. 

In the fine-tuning, we perform room classification and adjacency relationship classification tasks on the topological graph as reconstruction tasks: room classification forces room embeddings to contain their category information, while adjacency relationship classification forces room embeddings to contain the correct adjacency relationships. As a result, room embeddings effectively encode the entire information of the topological graph, serving as a constraint for node generation. Both room classification and adjacency relationship classification use the cross-entropy loss. The network architecture of the topology Transformer is shown in Figure 7. We fine-tune using the real topological graphs from the training dataset. 

## **Topology-constrained generation** 

The topological graph is defined as an undirected graph _G_ top = ( _V_ top _, E_ top), where each vertex _v_ top _,i_ = ( _ri_ ) _∈ V_ top represents a room, and a one-hot encoded vector _ri ∈ {_ 0 _,_ 1 _}_[7] indicates the room category. The edge ( _v_ top _,i, v_ top _,j_ ) _∈ E_ top represents an adjacency relationship between a pair of rooms (i.e., whether they share a wall). Considering the graph-like nature of this problem, we use the topology Transformer to encode it. 

## **Experimental setup** 

The implementation details of the different networks are as follows, with hyperparameters chosen based on empirical observations and performance on the validation dataset. 

## **Unconstrained generation** 

**Node Transformer** The node Transformer consists of 24 layers with an embedding dimension of 256, resulting in a total of 19 million parameters. The batch size is set to 256, and the optimizer is Adam (Kingma 2014). The training is conducted for 1,000,000 steps. The initial learning rate is 1 _×_ 10 _[−]_[4] , which is reduced by a factor of 0.1 after 500,000 steps. 

In the pre-training, we pre-compute room counts (ranging from 4 to 8), adjacency relationships (“True” or “False”), and room categories from the dataset. We then randomly sample the room count, adjacency relationships, and room categories to construct random topological graphs, which are used as input to the topology Transformer. Since the room count in the topological graph is uncertain, padding is required, and a cross-attention mask is used to limit attention to all nodes and real rooms. The input to the topology 

**Edge Transformer** The edge Transformer consists of 12 layers with an embedding dimension of 256, resulting in a 

total of 10 million parameters. The batch size is set to 8. Given the significant impact of edge prediction quality on the results, we implement a learning rate decay strategy for training: the initial learning rate is 1 _×_ 10 _[−]_[4] , and the optimizer is Adam (Kingma 2014). The performance on the validation dataset is closely monitored, and if the validation metric shows no improvement for five consecutive evaluations, the learning rate is reduced by a factor of 0.1. If there is no improvement for 20 consecutive evaluations, training is terminated, and the model with the best performance on the validation dataset is selected. Evaluation is performed every 1,000 steps. The performance of the edge prediction model peaked at 61,000 steps. 

## **Constrained generation** 

For the constrained generation, we increased the embedding dimension of the node Transformer to 512, resulting in a total of 96 million parameters. This change is made as we have observed that using the same 19 million parameters, adding topological constraints could reduce performance, possibly due to the difficulty of accommodating multiple types of information such as topological graphs and coordinates within the 256-dimensional space. The boundary constraints are configured similarly. The configuration of the Edge Transformer remained unchanged. 

**Boundary CNN** The boundary CNN has 31 million parameters. During the pre-training phase, the batch size is set to 16, and the optimizer is Adam (Kingma 2014) with an initial learning rate of 1 _×_ 10 _[−]_[4] . We use the same learning rate decay strategy for training as the edge Transformer. The performance peaked at 5,000 steps during pre-training. For fine-tuning, we restarted the learning rate at 1 _×_ 10 _[−]_[4] with a batch size of 16, and evaluation was performed every 100 steps, continuing training until 6,700 steps. 

**Topology Transformer** The topology Transformer consists of 24 layers with an embedding dimension of 256, resulting in a total of 19 million parameters. During the pretraining phase, the batch size is set to 2048, and the optimizer is Adam (Kingma 2014) with an initial learning rate of 1 _×_ 10 _[−]_[4] . We use the same learning rate decay strategy for training as the edge Transformer. The performance peaked at 11,000 steps during pre-training. For fine-tuning, we restarted the learning rate at 1 _×_ 10 _[−]_[4] with a batch size of 256, continuing training for 6,000 steps with the same method. 

## **Dataset** 

We first extract the 2D coordinates of wall junctions and segments. The image is binarized, with the walls represented as white and all other areas as black. Due to varying wall thicknesses, we regularize the thickness through repeated morphological operations (erosion, dilation) and template matching. We iteratively erode the white pixels representing the wall junction components until the next erosion step results in a decrease in the number of connected components in the image. This indicates that some wall segments have been eroded to a thickness of 1 pixel. 

Next, we apply a series of template matching operations, sliding a 3×3 window across the image at this stage. If a match is successful, the matching area (the 3×3 window) is marked as white. The templates represent lines with a thickness of 1 pixel or local wall shapes that are defective. Through this process, the shape of the walls is gradually standardized, and the thickness becomes more uniform. This iterative process continues until the erosion reduces the thickness of all walls to 1 pixel; at this point, further erosion would cause all walls to disappear, turning the image completely black. At this stage, the wall structure is what we require. 

Finally, the wall junctions and segments are extracted from the image, where the wall thickness has been uniformly reduced to 1 pixel. We obtain semantics from the four-channel images of the original RPLAN dataset. Any images that fail to process at this step are discarded. We obtain 71,763 vectorized floorplan images, randomly splitting them into 3,000 for the validation set, 3,000 for the test set, and the remainder for the training set. In the original RPLAN dataset, rooms are divided into 14 categories, which we merged into 7 categories: _Living room_ , _Bedroom_ , _Kitchen_ , _Bathroom_ , _Balcony_ , _Storage_ , _External area_ . 

Figure 8 (b1) shows some floorplan samples obtained from the above process. The RPLAN dataset (Wu et al. 2019) comes from real residential layouts, which do not contain slanted walls. To verify that our method is also applicable to floorplans with slanted walls, we heuristically deform the “peninsula-like” rectangular balcony which surrounded by the _External area_ on three sides into an isosceles trapezoid with the top base being 0.618 times the length of the bottom base. The data after the slanting deformation is shown in Figure 8 (b2). 

We have conducted distributional statistics (Figure 8(a)) on the processed RPLAN dataset (Wu et al. 2019), including the number of wall junctions, the number of wall segments, the number of rooms, and the quantity of each room category. 

## **More results** 

Figure 9 displays a comparison of boundary-constrained generation across different methods, including the groundtruth (GT), _Graph2Plan_ (Hu et al. 2020), _WallPlan_ (Sun et al. 2022), and ours. Figure 10 provides a comparison of topology-constrained generation among various techniques. including _HouseDiffsion_ (Shabani, Hosseini, and Furukawa 2023), _House-GAN++_ (Nauata et al. 2021) and ours. Figure 11 showcases the results of unconstrained generation. Figure 12 showcases the results of unconstrained generation with slanted walls. Figures 13 and 14 present the results of boundary-constrained generation by our method, while Figures 15 and 16 illustrate the results of topology-constrained generation by our method. 

## **References** 

Chen, J.; Zhang, R.; Zhou, Y.; and Chen, C. 2024. Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints. _arXiv preprint arXiv:2402.04754_ . 

**==> picture [357 x 124] intentionally omitted <==**

**----- Start of picture text -----**<br>
 Type Number of Rooms Number of Corners<br>(a1) (a2) (a3)<br>Ce]<br> of Edges 1<br>;<br>I<br>I<br>I<br>!1<br>f]<br>!<br>I<br>3840 4244464850 54 60 !<br>of Edges 1<br>(a4) I (b1) (b2)<br>**----- End of picture text -----**<br>


Figure 8: Dataset. (a1)-(a4): distributional statistics on the processed RPLAN dataset (Wu et al. 2019), including the quantity of each room category (a1), the number of rooms (a2), the number of wall junctions (a3) and the number of wall segments (a4). (b1)-(b2): some data samples obtained from the above data processing, (b1) shows original floorplans, and (b2) shows floorplans with slanted walls. 

Hu, R.; Huang, Z.; Tang, Y.; Van Kaick, O.; Zhang, H.; and Huang, H. 2020. Graph2plan: Learning floorplan generation from layout graphs. _ACM Transactions on Graphics (TOG)_ , 39(4): 118–1. 

Kingma, D. 2014. Adam: a method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ . 

Li, J.; Yang, J.; Zhang, J.; Liu, C.; Wang, C.; and Xu, T. 2020. Attribute-conditioned layout gan for automatic graphic design. _IEEE Transactions on Visualization and Computer Graphics_ , 27(10): 4039–4048. 

Nauata, N.; Hosseini, S.; Chang, K.-H.; Chu, H.; Cheng, C.Y.; and Furukawa, Y. 2021. House-gan++: Generative adversarial layout refinement network towards intelligent computational agent for professional architects. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 13632–13641. 

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L. u.; and Polosukhin, I. 2017. Attention is All you Need. In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.; Vishwanathan, S.; and Garnett, R., eds., _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc. 

Wu, W.; Fu, X.-M.; Tang, R.; Wang, Y.; Qi, Y.-H.; and Liu, L. 2019. Data-driven interior plan generation for residential buildings. _ACM Transactions on Graphics (TOG)_ , 38(6): 1–12. 

Zhang, H.; Cisse, M.; Dauphin, Y.; and Lopez-Paz, D. 2017. mixup: Beyond Empirical Risk Minimization. 

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In _Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18_ , 234–241. Springer. 

Shabani, M. A.; Hosseini, S.; and Furukawa, Y. 2023. Housediffusion: Vector floorplan generation via a diffusion model with discrete and continuous denoising. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 5466–5475. 

Sun, J.; Wu, W.; Liu, L.; Min, W.; Zhang, G.; and Zheng, L. 2022. Wallplan: synthesizing floorplans by learning to generate wall graphs. _ACM Transactions on Graphics (TOG)_ , 41(4): 1–14. 

Input boundary GT _Graph2Plan WallPlan_ Ours 

Figure 9: More on the comparison of boundary-constrained generation across different methods. 

Input topology _HouseDiffsion House-GAN++_ Ours 

Figure 10: More on the comparison of topology-constrained generation among various techniques. 

Figure 11: More results of unconstrained generation by our method. 

Figure 12: More results of unconstrained generation with slanted walls by our method. 

Figure 13: More results of boundary-constrained generation by our method: part (I). 

eh ae ae et a ee a Figure 14: More results of boundary-constrained generation by our method: part (II). 

Figure 15: More results of topology-constrained generation by our method: part (I). 

Figure 16: More results of topology-constrained generation by our method: part (II). 

