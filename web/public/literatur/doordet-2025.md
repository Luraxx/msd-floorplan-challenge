# DoorDet: Semi-Automated Multi-Class Door Detection Dataset via Object Detection and Large Language Models 

Licheng Zhang[1*] , Bach Le[1] , Naveed Akhtar[1] , Tuan Ngo[1] 

> 1The University of Melbourne, Parkville, Melbourne, 3010, Victoria, Australia. 

*Corresponding author(s). E-mail(s): licheng.zhang@student.unimelb.edu.au; Contributing authors: bach.le@unimelb.edu.au; naveed.akhtar1@unimelb.edu.au; dtngo@unimelb.edu.au; 

## **Abstract** 

Accurate detection and classification of diverse door types in floor plans drawings is critical for multiple applications, such as building compliance checking, and indoor scene understanding. Despite their importance, publicly available datasets specifically designed for fine-grained multi-class door detection remain scarce. In this work, we present a semi-automated pipeline that leverages a state-of-theart object detector and a large language model (LLM) to construct a multi-class door detection dataset with minimal manual effort. Doors are first detected as a unified category using a deep object detection model. Next, an LLM classifies each detected instance based on its visual and contextual features. Finally, a human-in-the-loop stage ensures high-quality labels and bounding boxes. Our method significantly reduces annotation cost while producing a dataset suitable for benchmarking neural models in floor plan analysis. This work demonstrates the potential of combining deep learning and multimodal reasoning for efficient dataset construction in complex real-world domains. 

**Keywords:** Multi-class Door Detection, Benchmark Dataset, Large Language Models, Co-DETR 

1 

## **1 Introduction** 

Architectural floor plan drawings are fundamental to the design, analysis, and regulation of building layouts, and they are widely used in both traditional computer-aided design (CAD) and modern architectural workflows [1]. Automatic interpretation of these drawings has attracted increasing interest in the computer vision and machine learning communities due to its broad range of applications, including building information modeling (BIM) [2], building compliance checking (BCC) [3], and automated spatial analysis [4]. Among the key structural elements in floor plans, doors play a critical role, serving not only as physical connections between spaces but also as indicators of accessibility, circulation patterns, and functional zoning within a building [5]. Table 1 summarizes the presence of door categories across various datasets, showing that the majority include door annotations, which underscores the critical role of doors in floor plan interpretation. 

Although most existing floor plan datasets include the door category, they typically treat all doors as a single class. While some datasets [6, 7] provide annotations for different door types, these distinctions are largely structural (e.g., single, double, sliding) rather than functional, and thus do not meet the requirements of our task. 

Knowing the functional type of each door offers several benefits. Primarily, it aids in building compliance checking and enhances indoor scene understanding. For instance, fire safety is a fundamental aspect of modern architectural design worldwide, aimed at protecting human life and property [8]. At its core, the provision and placement of emergency exit doors play a pivotal role in emergency route planning, accessibility assessment, and automated building code compliance checking. Additionally, having a multi-class door detection dataset enables us to evaluate different object detection methods and determine whether they work effectively for this specific task and domain. As a result, there is an increasing need for a publicly available dataset with highquality, fine-grained annotations of diverse door types. 

To create an object detection dataset specifically for multi-class door detection, a straightforward approach is to manually annotate each image from scratch. This requires annotators to draw bounding boxes that precisely align with the boundaries of each door and to carefully analyze the surrounding context to determine the appropriate category. However, this manual process is both time-consuming and laborintensive. To overcome these challenges, in this paper, we propose a novel dataset construction approach that significantly reduces the need for human intervention. 

Recent advancements in deep learning-based object detection have yielded impressive results across diverse domains, including remote sensing [9, 10], 3D object detection [11, 12], small object detection [13–15], salient object detection [16, 17], etc. Concurrently, large language models (LLMs) or vision-language models (VLMs), such as GPT-4o [18], have demonstrated strong capabilities in understanding semantics, context, and structural relationships within complex data, including 3D CAD models [19]. These emerging technologies open new avenues for automatically identifying key architectural components and reasoning about their functional roles. 

Specifically, we leverage a state-of-the-art object detector to ensure high-precision door localization. In addition, a state-of-the-art LLM is employed to identify different door types in architectural floor plans. Our system is designed to detect all door 

2 

**Table 1** : Summary of door category availability in existing floor plan datasets. ‘BBox’ refers to bounding box. ‘M’ is short for million. 

|**Dataset**|**Door Availability**|**BBox Availability**|**Diferent Types**|**Functional**|**Year**|**Size**|**Public**|
|---|---|---|---|---|---|---|---|
|||||||||
|SESYD [20]<br>FPLAN-POLY [21]<br>LIFULL HOME’S [22]<br>CVC-FP [23]<br>Rent3D [24]<br>SydneyHouse [25]<br>R-FP500 [26]<br>Raster-to-Vector [27]<br>ROBIN [28]<br>RPLAN [29]<br>BRIDGE [30]<br>R2V [4]<br>R3D [4]<br>CubiCasa5K [1]<br>SCUT-AutoALP [31]<br>Rent3D++ [32]<br>ZSCVFP [33]<br>RuralHomeData [34]<br>RFP [35]<br>Kim et al. [36]<br>RUB [37]<br>FloorPlanCAD [6]<br>David et al. [7]<br>MLSTRUCT-FP [38]<br>Park et al. [39]<br>MSD [40]<br>ArchCAD-400K [41]<br>**DoorDet** (**Ours**)|✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|✓<br>✗<br>✗<br>✗<br>✓<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✓<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✗<br>✗<br>✓<br>✓|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✓<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✓<br>✓<br>✗<br>✗<br>✗<br>✗<br>✓|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✓|2010<br>2010<br>2015<br>2015<br>2015<br>2016<br>2017<br>2017<br>2017<br>2019<br>2019<br>2019<br>2019<br>2019<br>2020<br>2021<br>2021<br>2021<br>2021<br>2021<br>2021<br>2021<br>2023<br>2023<br>2024<br>2024<br>2025<br>2025|1000<br>42<br>5.33 M<br>122<br>215<br>174<br>500<br>870<br>510<br>80000<br>13000<br>815<br>232<br>5000<br>602<br>215<br>10800<br>800<br>7000<br>230<br>303<br>15663<br>35000<br>954<br>10000<br>5372<br>413062<br>4991|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✗<br>✗<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✗<br>✓<br>✓<br>✓|



instances, and subsequently distinguish them among various types. This prediction process is informed not only by the visual characteristics of door symbols but also by contextual cues, such as their spatial connectivity to rooms and the presence of relevant text annotations. Afterward, a human-in-the-loop refinement process is introduced, enabling domain experts to review and correct the automated predictions. 

As a key outcome of our work, we have successfully constructed a curated dataset comprising real-world architectural floor plans annotated with detailed door type labels. Unlike general architectural datasets that focus on broader elements such as rooms and walls, our dataset emphasizes fine-grained classification of door instances based on their functions. We posit that this dataset will play a crucial role in advancing research in building code compliance automation, indoor semantic analysis, and object detection. The dataset will be made publicly available upon acceptance. To sum up, our contributions include the following. 

- We propose a novel methodology for object detection dataset construction, which we argue generalizes well to a broad range of related tasks. Compared to conventional dataset generation approaches, our method significantly reduces the need for manual effort. 

- We construct a curated dataset, **DoorDet** , consisting of annotated door instances derived from real-world floor plans. The dataset features fine-grained labels that distinguish multiple door categories. To the best of our knowledge, this is the first object detection dataset to include detailed door types categorized by functional roles. 

- We benchmark **DoorDet** using multiple state-of-the-art object detection models, demonstrating its utility for both training and evaluation in object detection tasks. 

3 

- We demonstrate the cross-domain generalizability of models trained on **DoorDet** , showing effective adaptation across various architectural layout datasets. 

## **2 Related Work** 

## **2.1 Object Detection in Floor Plans** 

Object detection in floor plans is an active research direction. Xu et al. [42] employed a multi-scale detection strategy with multiple detection heads to enhance the detection of small and overlapping objects in floor plans. Likewise, Shehzadi et al. [43] introduced a semi-supervised learning framework based on Mask R-CNN [44] enhanced with a student-teacher architecture, which effectively learnt from a small fraction of labeled data supplemented with a larger set of unlabeled data. Similarly, Mishra et al. [45] utilized Cascade Mask R-CNN [46], integrating deformable convolution to better capture geometric variations of objects in floor plans. Recently, Jakubik et al. [47] have identified uncertain symbols in floor plans and actively queried human annotators for clarification. Building upon earlier efforts, Lv et al. [35] employed yolov4 [48] with Mosaic data augmentation to detect and group text and symbols on floor plans, improving semantic context and accuracy in identifying room types. In another related approach, Lu et al. [34] simultaneously recognized graphical elements and detected room-type texts, addressing the intertwined nature of structural and semantic information in floor plans. 

Additionally, in [49], the object detection component of 3DPlanNet was designed to identify architectural elements such as walls, doors, windows, and rooms within 2D floor plan images. Other efforts, such as those by Surikov et al. [50], detected architectural elements such as doors, windows, and furniture using Faster R-CNN [51]. In a related study, Sch¨onfelder et al. [52] focused on detecting textual regions in architectural floor plans and evaluated multiple object detection methods: yolov5 [53], yolov7 [54], yolov8 [55], yolor [56], and Faster R-CNN [51]. In another line of work, Oh et al. [57] analyzed construction site images and architectural drawings to identify and localize elements related to finishing works, such as masonry and tiling, using yolov5 [53]. Another relevant contribution was by Park et al. [39], who utilized yolov3 [58] for detecting architectural objects in floor plan images, achieving high accuracy by training on a large, richly annotated dataset of spatial layouts. 

## **2.2 Datasets for Floor Plan Analysis** 

Delalandre et al. [20] developed the SESYD dataset, a synthetic collection for benchmarking symbol recognition and spotting systems. Similarly, Rusi˜nol et al. [21] gathered 42 high-resolution architectural floor plan images from four architectural projects, annotated with 38 symbol classes. Further, the dataset in [23] consisted of 122 scanned floor plan documents with ground-truth annotations for structural symbols like rooms, walls, doors, windows, parking doors, and separations. In another line of work, in [24], the Rent3D dataset consisted of 1570 images captured from 215 apartments, each annotated with detailed floor-plan information including walls, doors, and 

4 

windows. It provided ground truth 3D layout data and floor-plan priors to facilitate research on monocular indoor layout estimation and scene understanding. Extending this effort, Chu et al. [25] introduced the SydneyHouse dataset, comprising 174 residential buildings annotated with floor plan alignment, 3D structure, and textures, where 3D annotations included height, window/door locations, and pose. In a related contribution, Dodge et al. [26] presented the R-FP500 dataset with 500 floor plan images collected from real estate websites, manually annotated for walls, architectural objects, and text labels. Additionally, Liu et al. [27] annotated 870 floor plan images with walls, doors, icons, and junctions. Meanwhile, Sharma et al. [28] created the ROBIN dataset containing 510 real floor plan images with detailed room and furniture symbol annotations. 

On a larger scale, Wu et al. [29] collected real residential floor plans from a Chinese real estate website and automatically parsed and annotated room locations, shapes, and types using a combination of heuristic rules and pre-trained models, resulting in over 80000 floor plans across 21 room types. Similarly, Goyal et al. [30] introduced the BRIDGE dataset, sourced from architectural websites and public sources, annotated with bounding boxes and labels for common architectural elements like doors, windows, and furniture. Along the same line, Zeng et al. [4] prepared two datasets: R2V and R3D, which included floor plan images with annotations of room boundaries and types, supplemented by additional floor plan images. Moreover, Kalervo et al. [1] annotated 5000 floor plan images with polygon-level labels for rooms, walls, doors, windows, furniture, and fixtures, covering over 80 object classes. 

Continuing this trend, Liu et al. [31] developed the SCUT-AutoALP dataset containing 602 annotated samples, divided into residential floor plans (300 images with layout, boundary, and attribute labels) and urban campus plans (302 images with the same annotations). To extend prior datasets, Vidanapathirana et al. [32] introduced Rent3D++, an augmented dataset extending Rent3D [24], which included 215 floor plans from rental listings, 1570 room photographs, annotated with room types, walls, doors, windows, and real-world scale and layout information. Similarly, Dong et al. [33] developed a private dataset with 10800 color floor plan images encompassing a variety of decorative textures and styles. In parallel, Lu et al. [34] constructed a dataset of 800 real-world floor plan images from Chinese rural residences annotated with geometric components such as walls, doors and windows, as well as semantic labels including room types like bedrooms, kitchens, and storage rooms, and associated textual annotations extracted from the plans. 

Also, Lv et al. [35] introduced a dataset with around 7000 annotated instances of rooms, doors, walls, windows, and furniture from residential floor plans, featuring detailed labels of room types and architectural elements to support floor plan recognition and reconstruction. Further, the dataset in [36] featured floor plan images of large-scale complex buildings, where a scale matching module handled varying scales across floor plans, and each plan was divided into uniform patches labeled with object classes such as walls, doors, windows, elevators, and stairwells. From another perspective, Simonsen et al. [37] converted CAD files into graphs with nodes representing geometric primitives and edges encoding their topological and geometric relationships. 

5 

In addition, the FloorPlanCAD dataset [6] included over 15000 CAD floor plans preserving precise geometric and semantic information, with fine-grained annotations for 30 object categories such as stairs, furniture, doors, windows, and walls. Although the dataset included door locations and distinguished between basic door types, such as single, double, and sliding doors, these categories were simple and did not convey the functional role of the doors. 

More recently, David et al. [7] generated a dataset of 35000 annotated door images extracted from CubiCasa5K [1], focusing specifically on door identification and classification, alongside a literature review of key floor plan datasets. The dataset provided both door positions and types but the classification was fairly simple, concentrating on hinge side including left or right, opening direction like inward outward or swinging, and door style such as single, double, sliding or folding. Similarly, in [38], the MLSTRUCT-FP dataset included 954 floor plan images used for training and evaluating wall segmentation models, with over 1.3 million image patches extracted to capture detailed structural features. In a large-scale web mining effort, Park et al. [39] collected a large dataset of 100000 data rows through extensive web crawling, categorized into 21 classes and three spatial relationship categories. Furthermore, in [40], van Engelenburg et al. introduced the MSD dataset, comprising 5372 floor plans of medium to large-scale building complexes, encompassing over 18900 distinct apartments. Each floor plan was available in image, vectorized, and graph-based formats. Lastly, ArchCAD-400K [41] was a large-scale, publicly available dataset containing over 413000 annotated chunks from 5538 architectural CAD floor plans. Leveraging CAD layers and blocks, it automated detailed annotations across 27 categories including doors and structural elements, covering mostly commercial and public buildings. This dataset provided a valuable benchmark for panoptic symbol spotting and architectural CAD analysis. 

## **3 Methodology** 

In this paper, we propose a novel human-in-the-loop-assisted, vision-language-driven approach for constructing a multi-class door detection dataset, as illustrated in Figure 1. The pipeline begins with a state-of-the-art object detector that identifies all door instances as a single class. Subsequently, heuristic techniques are applied to facilitate downstream processing. A state-of-the-art LLM is then employed to both locate and predict the type of individual door instances. Finally, a human-in-the-loop refinement stage is introduced to transform the initial coarse annotations into high-quality, fine-grained labels. Our approach enables the efficient construction of a practically valuable dataset with significantly reduced manual effort and time cost, compared to conventional dataset creation methods. 

## **3.1 Single-Class Door Detection** 

Benefiting from existing publicly available single-class door detection datasets, we can effortlessly train a unified model that detects all door types as a single class, applicable across diverse real-world CAD drawings. To ensure high accuracy in door detection on out-of-domain data, we adopt a state-of-the-art object detector that 

6 

**Fig. 1** : Our proposed vision-language-driven human-in-the-loop pipeline for multiclass door detection dataset generation. 

achieves top-tier performance on the COCO benchmark [59], which is Co-DETR [60]. Co-DETR exhibits strong performance in detecting small and dense objects, making it particularly suitable for floor plan scenarios where door symbols are often compact and tightly clustered. Accordingly, we adopt Co-DETR as our object detector. Compared to its predecessor DETR [61], which suffers from slow convergence and limited performance on dense or small object detection due to its merely one-to-one label assignment strategy, Co-DETR introduces a collaborative hybrid assignment mechanism, which combines one-to-one and many-to-one assignments during training, significantly improving both convergence speed and detection accuracy. In Co-DETR, the primary decoder performs one-to-one matching (as in DETR), used during both training and inference. In parallel, the auxiliary decoder adopts many-to-one matching (as in Faster R-CNN), used only during training and discarded at inference time. Both decoders share a common backbone and encoder but differ in their attention weights and matching strategies. Compared with DETR, Co-DETR achieves up to 2 _×_ faster convergence and improved performance on challenging detection tasks. 

Mathematically, given an input image **I** , the detector outputs a set of _N_ predicted door instances: 

**==> picture [248 x 13] intentionally omitted <==**

where _bi_ and _ci_ denote the bounding box and confidence score of door _i_ , respectively. The detection network is trained to detect only a single object category, in this case doors, by distinguishing between background and door instances. 

## **3.2 Door Type Prediction via LLMs** 

## **3.2.1 Motivation** 

Although object detectors effectively localize doors, they generally lack the semantic and contextual reasoning capabilities required to distinguish between functional door types. To address this limitation, we propose leveraging a vision-capable LLM 

7 

to infer door functionality directly from visual and contextual cues. Recent advancements in LLMs with vision capabilities, such as GPT-4o [18], enable the interpretation of images, diagrams, and floor plans in conjunction with textual information [62]. Their ability to integrate visual and semantic context makes them well-suited for highlevel interpretation of functional elements within architectural layouts. GPT-4.1 [63], one of OpenAI’s latest flagship models, supports multimodal inputs and introduces a one-million-token context window, significantly enhancing its capacity to reason over complex documents and large-scale visual layouts. Consequently, we adopt GPT-4.1 as the LLM in this work. 

## **3.2.2 Door Type Prediction** 

After obtaining the generic door detection model described in Section 3.1, we first perform inference on our selected floor plan images using the optimized model. This phase locates all door instances within each image. Subsequently, we process each detected door individually and assign a specific type to it. 

For each detected door instance, although we have its position within the image, LLMs are not well-suited to directly infer coordinates from positional values. Therefore, we provide the LLM with both the cropped door region and the full floor plan image, and then ask it to locate the corresponding region within the full floor plan. 

However, during implementation, we observed that providing only the cropped door region makes it difficult for the LLM to accurately identify the specific door, due to the similar appearance of doors and the limited contextual cues. To address this, we expand the cropped region by extending the bounding box with a fixed margin _m_ pixels on all sides (e.g., _m_ = 100) to include more surrounding context. Additionally, since the expanded region may contain multiple doors, we explicitly instruct the LLM to focus only on the central door within the cropped image. 

Formally, door localization via LLM can be expressed as: 

**==> picture [224 x 11] intentionally omitted <==**

where _Ci_ is the expanded cropped image patch of the _i_ -th detected door extracted from the full floor plan **I** , and _Li_ denotes the estimated location information of the door (e.g., room reference or spatial description within **I** ). 

After locating the door within the complete floor plan, we then prompt the LLM to identify the door type by analyzing its connectivity to surrounding rooms and leveraging semantic and contextual cues present in the layout. 

Analytically, the door type prediction can be formulated as: 

**==> picture [232 x 13] intentionally omitted <==**

where _Li_ denotes the location of the _i_ -th detected door. **I** denotes the full floor plan image. _P_ denotes the prompt including the task instruction, the door category definitions, as well as the output format. _Ti_ denotes the predicted door category. The LLM 

8 

integrates all these information and outputs the door type according to the rules outlined in _P_ . The output format is a category number or a combination (e.g., emergency exit combined with main entry) as specified. 

## **3.3 Human-in-the-Loop Refinement** 

Inevitably, LLM predictions cannot achieve perfect accuracy, and some degree of error is expected within the resulting dataset. To address this, we introduce a human-inthe-loop refinement stage, where trained annotators systematically verify and refine the coarse annotations produced by the model. We instruct annotators to correct mislabeled instances, add missing detections, and remove false positives. Additionally, they adjust bounding box positions when the original boxes fail to fully enclose the door or include excessive non-door regions. This refinement process is substantially faster than annotating from scratch, which not only requires drawing bounding boxes but also demands careful consideration of the correct category for each instance, a task that is both time-consuming and labor-intensive. By starting with model-generated predictions and applying targeted corrections, we greatly reduce the manual workload while still ensuring high-quality annotations. 

To assess the impact of introducing the human-in-the-loop mechanism, we analyze the correlation between task difficulty and the presence of human intervention within our proposed framework for multi-class door detection. 

## _**Task Formulation**_ 

Let _T_ denote the set of detection tasks, where each task corresponds to detecting a specific door type. For each task _t ∈T_ , we define _D_ ( _t_ ) _∈_ R _≥_ 0 as the _task difficulty_ , and ∆( _t_ ) _∈_ R as the _performance gain_ resulting from human intervention. 

Our goal targets at quantifying the correlation between _D_ ( _t_ ) and ∆( _t_ ): Corr _t∈T_ � _D_ ( _t_ ) _,_ ∆( _t_ )�. A larger positive value indicates a stronger correlation between them. 

We define the task difficulty _D_ ( _t_ ) as the initial detection error obtained by the model before human intervention: 

**==> picture [258 x 12] intentionally omitted <==**

where ‘model-only’ indicates only relying on models, such as object detector and LLM, while without human-in-the-loop refinement. The lower the accuracy gets, the harder the task becomes. 

The performance gain ∆( _t_ ) is measured as the improvement in accuracy resulting from human intervention: 

**==> picture [291 x 13] intentionally omitted <==**

where ‘HITL’ denotes including the human-in-the-loop setup. A larger value of ∆( _t_ ) indicates a greater contribution of human feedback to performance improvement. 

9 

The above formulations enable us to analyze how the effectiveness of human feedback correlates with the inherent difficulty of each task in our multi-class door detection framework. 

## **4 DoorDet: Proposed Multi-Class Door Detection Dataset** 

Our dataset, **DoorDet** , is constructed from the CubiCasa5K dataset [1], which provides a diverse collection of floor plans encompassing a wide range of architectural styles and spatial layouts. The dataset consists of 5000 samples from real residential buildings across multiple countries, divided into 4200 training, 400 validation, and 400 test instances. These samples represent multi-room and multi-floor structures, including both apartments and standalone houses. Many floor plans include textual annotations indicating room types such as bedrooms, bathrooms, and kitchens, along with standardized architectural symbols for doors and other structural elements. The images feature minimal decorative content and maintain consistent symbology. The embedded text within the plans is particularly valuable for inferring room functions, which can in turn facilitate the prediction of corresponding door types. 

We first extract floor plan images from the source SVG files in the CubiCasa5K dataset, rather than relying on the pre-rendered raster images provided, and retain the native scale and precision of the floor plans. The annotations are then generated using our proposed method described in Section 3, which automatically produces coarse annotations via Co-DETR and GPT-4.1. These are subsequently refined through a human-in-the-loop process that corrects mispredictions, removes false positives, and results in high-quality, fine-grained annotations for each door instance. 

For the first step, which is to train a single-class door detection model, we construct a larger dataset by combining two publicly available single-class door detection datasets, which contain 1047 and 837 training examples, respectively [64, 65]. 

Our constructed **DoorDet** dataset includes a variety of door types which are detailed as follows. 

- **Main entry door:** This refers to the primary entrance to a building or unit, typically connecting interior spaces to the exterior or shared access areas. 

- **Bedroom door:** This type of door leads to sleeping areas within the interior of the building. 

- **Bathroom or washroom door:** These doors provide access to hygiene-related spaces such as toilets, bathrooms, or showers. 

- **Kitchen door:** A kitchen door connects the kitchen to adjacent spaces such as dining rooms, hallways, or utility areas. 

- **Living room or dining room door:** These doors lead to communal living spaces such as lounges, living rooms, or dining areas. 

- **Laundry or utility room door:** This type of door provides access to service areas such as laundry rooms, mechanical rooms, or utility closets. Any door that does not fit clearly into another category is typically assigned to this type. 

10 

**Table 2** : Basic information about the **DoorDet** dataset. ‘BBox’ refers to bounding box. 

|**Property**|**Statistic**|
|---|---|
|||
|Total foor plan images<br>Number of door categories<br>Average doors per image<br>Average image resolution<br>Annotation format<br>Source<br>Train/Val/Test split|4991<br>10<br>7.81<br>1495.2_×_1310.6<br>BBox + type<br>CubiCasa5K [1]<br>4192/400/399|



- **Garage door:** This is a large door that provides vehicle access to a garage, typically from the outside. Interior doors that allow pedestrian access between the garage and living spaces are also included in this category. 

- **Balcony or terrace door:** These are external doors that open to outdoor areas such as balconies, terraces, or verandas. 

- **Emergency exit door:** This door serves as a designated exit during emergencies and is essential for safety compliance and evacuation planning. 

- **Study room door:** This door provides access to private or semi-private workspaces such as studies, home offices, or libraries. The type is manually determined by referring to example patterns found during the refinement process. 

## **4.1 Data Statistics** 

To better understand the scope and richness of our dataset, we summarize its core characteristics below. Table 2 presents the key statistics, including the total number of floor plan images, the average number of doors per image, the average image resolution, and other important attributes. Each image contains multiple door instances spanning a diverse set of categories. Notably, the high image resolution sets **DoorDet** apart from most existing object detection benchmarks [59, 66]. Each door instance is annotated with both a category label and a bounding box. To the best of our knowledge, **DoorDet** is the first dataset in the architectural domain to provide such detailed annotations for door instances, encompassing both their locations and functional types. 

In addition to these global statistics, we also present the total number of door instances for each category in the dataset. Figure 2 illustrates the distribution of different door categories. As shown in the figure, the study room door and garage door categories contain significantly fewer instances compared to others, while all remaining categories have over 1000 instances, with laundry or utility room doors and emergency exit doors exceeding 10000 instances. This indicates a clear class imbalance within the dataset. Such imbalance is uncommon in conventional object detection benchmarks, making **DoorDet** a more challenging and realistic testbed for evaluating model robustness in imbalanced scenarios. 

11 

**==> picture [357 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
· 10 [4]<br>1 . 2 11 , 480<br>10 , 714<br>1<br>0 . 8<br>6 , 900 6 , 634<br>0 . 6<br>4 , 397<br>0 . 4<br>2 , 950<br>0 . 2 1 , 378 1 , 422<br>331 105<br>0<br>Door Category<br>Mainentry Bedroom orwashroom Kitchen diningroom utilityroom Garage orterrace exit Studyroom<br>or or Emergency<br>Balcony<br>room<br>Bathroom Laundry<br>Living<br>Instances<br>of<br>Number<br>**----- End of picture text -----**<br>


**Fig. 2** : Distribution of instances across the 10 door categories in the **DoorDet** dataset. 

Furthermore, we analyze the distribution of the number of door categories per image in Figure 3. As shown in the figure, most images contain around six types of door instances, which aligns with the nature of floor plan layouts, typically composed of diverse room types, each associated with function-specific doors, indicating the complexity and challenge of accurate door detection and fine-grained classification. 

Finally, we analyze the distribution of the number of door instances per image. Figure 4 shows how many images contain different numbers of annotated door instances. As observed, most images contain more than five doors, which highlights the practical significance and inherent difficulty of the door detection task in the dataset. 

## **5 Experiments** 

## **5.1 Datasets** 

As mentioned earlier, we use a combined dataset from [64, 65] to train a unified singleclass door detection model, with training performed on the merged training sets and evaluation on the merged test sets. 

Benchmark experiments are conducted on our constructed **DoorDet** dataset. We follow the original split settings from CubiCasa5K [1], using its training, validation, and test file lists to define the corresponding splits in our dataset. 

12 

**==> picture [368 x 186] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 , 042<br>2 , 000<br>1 , 500<br>1 , 000 932<br>782<br>500 407<br>194<br>61 99<br>15 7<br>0<br>1 2 3 4 5 6 7 8 9<br>Number of Door Categories per Image<br>Images<br>of<br>Number<br>**----- End of picture text -----**<br>


**Fig. 3** : Distribution of door categories per image in the **DoorDet** dataset. 

**==> picture [362 x 185] intentionally omitted <==**

**----- Start of picture text -----**<br>
563<br>542 532<br>447<br>400 367 368<br>302<br>284<br>210 208<br>200<br>139 130 118<br>73 67<br>35 47 37 32<br>19 19<br>0<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21+<br>Number of Door Instances per Image<br>Images<br>of<br>Number<br>**----- End of picture text -----**<br>


**Fig. 4** : Distribution of the door instances per image in the **DoorDet** dataset. 

## **5.2 Benchmarking Methods** 

**Co-DETR [60]:** For Co-DETR, we adopt the best-performing Co-DETR variant with a Vision Transformer Large (ViT-L) [67] backbone. It achieves a 66.0 mAP on the COCO `test-dev` . We utilize 300 learned object queries for it. The model is optimized using the AdamW optimizer with an initial learning rate of 1 _×_ 10 _[−]_[4] , which is decayed by a factor of 0.5 at epochs 5, 6, 7, and 10, respectively, and then kept fixed for the remainder of training, up to around 20 epochs, which is stopped once accuracy 

13 

plateaus. Batch size is set to 2. The experiments are performed using two NVIDIA A100 GPUs. The network is initialized with the pretrained weights of Co-DETR on the COCO dataset [59]. During inference, the object confidence threshold is set to 0.3. 

**InternImage-H [68]:** For InternImage, we adopt the InternImage-H backbone, which contains approximately 1 billion parameters and incorporates the novel DCNv3 deformable convolution operator. It achieves a 65.5 mAP on the COCO `test-dev` . The detection framework is based on DINO [69], and we apply multi-scale training to enhance performance. Optimization is performed using the AdamW optimizer with a starting learning rate of 1 _×_ 10 _[−]_[4] , combined with layer-wise learning rate decay and a weight decay of 1 _×_ 10 _[−]_[4] . Training is conducted for 45000 iterations, including a 500iteration warm-up phase. The learning rate is decayed by a factor of 0.1 at 20000 and 40000 iterations, respectively. The model is trained with a batch size of 1 using one NVIDIA A100 GPU. The network is initialized with pretrained InternImage-H weight on the COCO dataset [59]. 

**Focal-Stable-DINO [70]:** Focal-Stable-DINO combines the FocalNet-Huge backbone [71] with the Stable-DINO detection head [72], achieving 64.8 mAP on COCO `test-dev` without any test-time augmentation. FocalNet-Huge provides strong feature representation while maintaining computational efficiency. Stable-DINO introduces a position-supervised classification loss and a position-modulated matching cost, which together stabilize decoder-layer predictions and eliminate the unstable matching paths commonly found in standard DETR variants. The model is optimized using AdamW with a base learning rate of 1 _×_ 10 _[−]_[4] and a backbone-specific learning rate scaled by a factor of 0.1. The weight decay is set to 1 _×_ 10 _[−]_[4] . Training is conducted on two NVIDIA A100 GPUs with a batch size of 2, for a total of 120000 iterations. The learning rate is decayed by a factor of 0.1 at 82500 and 110000 iterations, respectively. The number of de-noising queries is set to 1000. The parameters of FocalNet-Huge are initialized with pretrained weights on ImageNet [73], obtained via the timm library [74]. 

**EVA [75]:** We employ an EVA-style Vision Transformer Giant (ViT-G) [76] backbone featuring 40 Transformer layers, each with 16 attention heads and an embedding dimension of 1408. The model processes inputs at a resolution of 1280 _×_ 1280 with patch size 16 and uses windowed attention with windows of size 16, interleaved with global attention every four blocks. Regularization includes a drop path rate of 0.6 and activation checkpointing to reduce memory usage. It achieves a 64.7 mAP on COCO `test-dev` . Training uses the AdamW optimizer with a base learning rate of 2 _._ 5 _×_ 10 _[−]_[5] , decayed layer-wise by a factor of 0.9 across the 40 layers. The learning rate is set to 1 _×_ 10 _[−]_[5] at iteration 35000 and further reduced to 1 _×_ 10 _[−]_[6] at iteration 40000, over a total of 45000 training steps, with a warmup of 500 iterations at the beginning. The batch size is set to 8. Training is conducted using two NVIDIA A100 GPUs. We initialize the model using weights pretrained on the COCO dataset [59]. 

## **5.3 Implementation details** 

For GPT-4.1, we access its capabilities through the OpenAI API. _m_ is set to 200. 

14 

**Table 3** : Single-class door detection results with Co-DETR on the combined dataset. 

|**Method**|**mAP**|**mAP@50**|
|---|---|---|
|Co-DETR (single-class)<br>0.403<br>0.793|||



## **5.4 Evaluation Metrics** 

We adopt mAP (mean Average Precision at IoU thresholds from 0.50 to 0.95), and mAP@50 (mean Average Precision at IoU = 0.50) to assess the performance of singleclass and multi-class door detection. Since the bounding boxes in our dataset do not precisely align with the exact boundaries of door instances, we additionally use mAP@50 as the evaluation metric to provide a more robust and tolerant measure of detection performance. 

## **5.5 Experiment Results** 

## **5.5.1 Single-class Door Detection** 

Single-class door detection serves as the foundation of our approach, owing to the availability of existing single-class door datasets. Table 3 reports the mAP and mAP@50 of Co-DETR on the combined test set from [64, 65], after being trained on their combined training set. As shown in Table 3, the model achieves a reasonably high mAP@50 of 0.793, indicating that it can reliably localize door instances with moderate IoU thresholds. However, the overall mAP score of 0.403 suggests that precise localization remains challenging under stricter thresholds. 

## **5.5.2 Multi-class Door Detection** 

Multi-class door detection is evaluated on our **DoorDet** dataset. We benchmark four state-of-the-art object detection methods: Co-DETR [60], InternImage-H [68], FocalStable-DINO [70], and EVA [75], which currently rank among the top-performing models on the COCO benchmark [59]. 

Table 4 presents the results of different methods, reporting both mAP and mAP@50 for each class. These results establish a baseline for future research on our **DoorDet** dataset and highlight the challenging nature of the multi-class door detection task. Notably, the performance rankings are consistent with those observed on the COCO benchmark, although the performance gaps among methods are more pronounced. Figure 5 further illustrates our dataset’s inherent complexity for multi-class door detection. 

## **5.6 Ablation Studies** 

_**LLM:**_ Without incorporating the LLM, door type cannot be performed automatically, and we must rely entirely on manual annotation. In this context, the benefit of using an LLM lies in significantly reducing annotation time by automating the initial process. We consider two manual alternatives: (1) labeling all door instances from scratch, 

15 

**Table 4** : Benchmarking results of different multi-class door detection methods on the **DoorDet** dataset. ‘CD’ is short for ‘Co-DETR’. ‘IIH’ is short for ‘InternImage-H’. ‘FSD’ is short for ‘Focal-Stable-DINO’. 

|**Category**|**Category**|**mAP**<br>CD<br>IIH<br>FSD<br>EVA|**mAP**<br>CD<br>IIH<br>FSD<br>EVA|**mAP**<br>CD<br>IIH<br>FSD<br>EVA|**mAP**<br>CD<br>IIH<br>FSD<br>EVA|**mAP**<br>CD<br>IIH<br>FSD<br>EVA|**mAP@50**<br>CD<br>IIH<br>FSD<br>EVA|**mAP@50**<br>CD<br>IIH<br>FSD<br>EVA|**mAP@50**<br>CD<br>IIH<br>FSD<br>EVA|**mAP@50**<br>CD<br>IIH<br>FSD<br>EVA|
|---|---|---|---|---|---|---|---|---|---|---|
|||CD|IIH|FSD|||CD|IIH|FSD||
||||||||||||
|Main entry door<br>Bedroom door<br>Bathroom or washroom door<br>Kitchen door<br>Living room or dining room door<br>Laundry or utility room door<br>Garage door<br>Balcony or terrace door<br>Emergency exit door<br>Study room door|**0.907**<br>**0.891**<br>**0.887**<br>**0.852**<br>**0.805**<br>**0.853**<br>**0.806**<br>0.841<br>**0.872**<br>0.598||0.902<br>0.883<br>0.879<br>0.808<br>0.797<br>0.841<br>0.729<br>0.834<br>0.870<br>**0.599**|0.887<br>0.861<br>0.868<br>0.814<br>0.755<br>0.806<br>0.705<br>**0.855**<br>0.858<br>0.409|0.882<br>0.851<br>0.848<br>0.763<br>0.774<br>0.813<br>0.781<br>0.814<br>0.551<br>0.458|**0.996**<br>**0.983**<br>**0.982**<br>**0.924**<br>**0.919**<br>**0.966**<br>0.885<br>**0.956**<br>0.988<br>0.665||0.991<br>0.979<br>0.979<br>0.890<br>0.916<br>0.961<br>0.809<br>0.948<br>**0.990**<br>**0.666**|0.989<br>0.966<br>0.979<br>0.899<br>0.880<br>0.943<br>0.864<br>0.954<br>0.987<br>0.501|0.975<br>0.955<br>0.966<br>0.844<br>0.898<br>0.941<br>**0.902**<br>0.927<br>0.664<br>0.500|
||||||||||||
|All|**0.831**||0.814|0.782|0.753|**0.926**||0.913|0.896|0.857|



and (2) performing door detection first, followed by human-in-the-loop refinement to assign door categories. 

To validate the benefits of employing an LLM, we further provide statistics on the time required for refinement and compare it with the time needed for the above two alternatives. We randomly selected six examples and annotated them using labelImg[1] for each of the three methods. Prior to annotation, we prepared the category list in the tool so that labels could be easily selected from the predefined options. Additionally, since the bounding boxes generated by our proposed pipeline do not perfectly align with object boundaries, we did not enforce absolute alignment when drawing new bounding boxes to ensure a fair comparison. Notably, since annotators may remember labels when annotating the same samples multiple times, we adopt a counterbalancing strategy to mitigate such effects. Specifically, we divide the six samples into three subsets, each containing two samples with comparable labeling difficulty. The three subsets are first annotated separately, each using one of the three annotation methods. For the first subset, annotation is performed from scratch, including manually drawing bounding boxes. For the second subset, we apply our proposed pipeline incorporating the LLM. For the third, we first convert all labels to a single category (“door”) and then refine them to their specific door types. After this initial phase, we proceed to annotate the remaining examples using each method. In total, all six samples are labeled using each approach. We record and average the time required for each method. This counterbalanced design minimizes the influence of memory or familiarity with the samples, ensuring a fair comparison of annotation efficiency. 

Table 5 compares the manual annotation time across three methods: labeling from scratch, door detection with human-in-the-loop refinement (without LLM), and our proposed pipeline incorporating an LLM. From Table 5, we observe that our 

> 1https://github.com/tzutalin/labelImg 

16 

**Fig. 5** : An example illustrating the challenges of multi-class door detection is shown in boxes (1), (2), and (3). In case (1), the door provides the only access to the room labeled ‘PUKUH’, so its type is inferred based on the category of ‘PUKUH’. Similarly, the type of door in (2) is determined by its connection to ‘PSH’, while in (3), the door type is inferred from the interior room it is part of. This demonstrates that the classification of a door may depend on the room it overlaps with in some cases, while in others, it relies on the adjacent room. Such variability in contextual cues makes the task inherently complex. However, the categories of all three doors were correctly predicted by GPT-4.1. 

**Table 5** : Per-Image Annotation Time: Manual Labeling vs. Human-in-the-Loop with and without LLM. 

|Manual labeling from scratch<br>Door Detection + Human-in-the-Loop<br>Door Detection + LLM + Human-in-the-Loop|97.5<br>70.0<br>**54.5**|
|---|---|



method significantly reduces annotation time and associated effort compared to labeling from scratch. Moreover, integrating the LLM further improves efficiency over the refinement-only approach. Notably, even without the LLM, combining door detection with human feedback still outperforms manual annotation in terms of speed. Owning to the remarkable capabilities of modern LLMs, our proposed dataset generation pipeline can be readily extended to other tasks where annotation time is limited but strong performance is still desired. For instance, if we have a large collection of raw samples such as those in FloorPlanCAD [6], our proposed pipeline can rapidly generate a domain-specific object detection dataset, enabling efficient construction within a limited time budget. 

17 

**Fig. 6** : Examples of door type prediction with GPT-4.1. Boxes except green are predicted by GPT-4.1. Green boxes are found by us that are wrongly predicted. 

Furthermore, to provide an intuitive understanding of the benefits brought by LLMs, we present several examples of model-generated results prior to human-inthe-loop refinement. Figure 6 presents these results. As shown, GPT-4.1 accurately predicts the door types in most cases. 

_**Human-in-the-Loop:**_ The human-in-the-loop mechanism offers two key benefits. First, it substantially reduces annotation time, as demonstrated in Table 5. Second, by incorporating human corrections into the pipeline, it has the potential to enhance the overall object detection performance. 

To validate this, we conduct additional experiments on Co-DETR using the coarse dataset prior to human-in-the-loop refinement and compare the detection performance before and after the human-in-the-loop process. Table 6 presents the comparative results. The performance improves significantly after refinement, attributable to the more accurate and consistent annotations introduced through human feedback. 

18 

**Table 6** : Effect of human-in-the-loop refinement on Co-DETR for multi-class door detection. 

|**Method**|**mAP**|**mAP@50**|
|---|---|---|
|Co-DETR (w/o refnement)<br>0.578<br>0.650<br>Co-DETR (w/ refnement)<br>**0.831**<br>**0.926**|||



**Table 7** : Per-category performance of Co-DETR on the **DoorDet** dataset, comparing results **with** and **without** human-in-the-loop refinement. 

|results **with** and **without** human|-|in-the-loop|in-the-loop|refnem|ent.||||||
|---|---|---|---|---|---|---|---|---|---|---|
|**Category**||**mAP**||_D_(_t_)|∆(_t_)||**mAP@50**||_D_(_t_)|∆(_t_)|
|||w/o|w/||||w/o|w/|||
||||||||||||
|Main entry door<br>Bedroom door<br>Bathroom or washroom door<br>Kitchen door<br>Living room or dining room door<br>Laundry or utility room door<br>Garage door<br>Balcony or terrace door<br>Emergency exit door<br>Study room door|0.769<br>0.813<br>0.741<br>0.734<br>0.276<br>0.633<br>0.328<br>0.688<br>0.802<br>0.000||0.907<br>0.891<br>0.887<br>0.852<br>0.805<br>0.853<br>0.806<br>0.841<br>0.872<br>0.598|0.231<br>0.187<br>0.259<br>0.266<br>0.724<br>0.367<br>0.672<br>0.312<br>0.198<br>1.000|0.138<br>0.078<br>0.146<br>0.118<br>0.529<br>0.220<br>0.478<br>0.153<br>0.070<br>0.598|0.848<br>0.894<br>0.822<br>0.796<br>0.319<br>0.723<br>0.370<br>0.801<br>0.928<br>0.000||0.996<br>0.983<br>0.982<br>0.924<br>0.919<br>0.966<br>0.885<br>0.956<br>0.988<br>0.665|0.152<br>0.106<br>0.178<br>0.204<br>0.681<br>0.277<br>0.630<br>0.199<br>0.072<br>1.000|0.148<br>0.089<br>0.160<br>0.128<br>0.600<br>0.243<br>0.515<br>0.155<br>0.060<br>0.665|
||||||||||||
|All|0.578||0.831|0.422|0.253|0.650||0.926|0.350|0.276|



## **5.7 Correlation Between Task Difficulty and Human Feedback** 

Table 7 presents the per-category performance of the Co-DETR model on the **DoorDet** dataset, both **with** and **without** human-in-the-loop refinement. Additionally, it reports the task difficulty _D_ ( _t_ ) and performance gain ∆( _t_ ), as defined in Equations 4 and 5, respectively. As shown in Table 4, we observe that _D_ ( _t_ ) and ∆( _t_ ) exhibit a positive correlation in most cases, indicating that categories with higher task difficulty generally benefit more from human-in-the-loop refinement. 

## **5.8 Illustrations** 

Figure 7 illustrates several examples from other domains tested using the CO-DETR model optimized on our proposed **DoorDet** dataset. As shown, our dataset enables the trained object detection model to generalize to other datasets to some extent. 

## **6 Limitations and Future Work** 

One limitation of our work lies in the fact that all bounding boxes in our dataset are algorithmically generated, and therefore only approximate rather than perfectly accurate. Nevertheless, we make efforts to refine obvious errors, particularly when bounding 

19 

**==> picture [359 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) FloorPlanCAD [6] (b) ROBIN [28] (c) CVC-FP [23] (d) SESYD [20]<br>**----- End of picture text -----**<br>


**Fig. 7** : We conduct inference with the optimized Co-DETR model trained on our proposed **DoorDet** dataset using several datasets from other domains. 

boxes significantly omit parts of the true door or noticeably deviate from object boundaries, manually correcting such substantial inaccuracies to improve overall quality. As shown in Figure 6, the bounding boxes generally provide sufficient coverage of the objects, typically extending only slightly beyond their true boundaries. To further alleviate the limitation, given the approximate nature of the annotations, we adopt mAP@50 as a more tolerant evaluation metric to accommodate minor localization inaccuracies, thereby providing a practical basis for benchmarking future methods. 

As part of our future work, a promising direction is to explore effective methods for improving object detection performance on our dataset, particularly by leveraging techniques that address class imbalance. Another possible direction is to develop domain adaptation techniques to extend the use of our dataset to other domains. 

## **7 Conclusion** 

In this paper, we propose a novel approach to construct a highly useful multi-class door detection dataset in the field of floor plan drawings with minimal effort. We first detect all doors as a unified category using a state-of-the-art object detector, and then leverage the reasoning capabilities of an LLM to estimate door types based on their visual features and contextual cues. We validate our approach through extensive experiments, which demonstrate the effectiveness of our proposed method and the usefulness of the **DoorDet** dataset in terms of both accuracy and annotation efficiency. 

20 

**Data availability** The dataset supporting the results of this study will be made publicly available upon acceptance of the manuscript. The data repository link and DOI will be provided at that time. 

## **Declarations** 

**Conflict of interest** The authors declare that they have no conflict of interest. 

## **References** 

- [1] Kalervo, A., Ylioinas, J., H¨aiki¨o, M., Karhu, A., Kannala, J.: Cubicasa5k: A dataset and an improved multi-task model for floorplan image analysis. In: Image Analysis: 21st Scandinavian Conference, SCIA 2019, Norrk¨oping, Sweden, June 11–13, 2019, Proceedings 21, pp. 28–40 (2019) 

- [2] Pizarro, P.N., Hitschfeld, N., Sipiran, I., Saavedra, J.M.: Automatic floor plan analysis and recognition. Automation in Construction **140** , 104348 (2022) 

- [3] Chen, D., Chen, L., Zhang, Y., Lin, S., Ye, M., Sølvsten, S.: Automated fire risk assessment and mitigation in building blueprints using computer vision and deep generative models. Advanced Engineering Informatics **62** , 102614 (2024) 

- [4] Zeng, Z., Li, X., Yu, Y.K., Fu, C.-W.: Deep floor plan recognition using a multi-task network with room-boundary-guided attention. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9096–9104 (2019) 

- [5] Zhang, C., Cui, Z., Zhang, Y., Zeng, B., Pollefeys, M., Liu, S.: Holistic 3d scene understanding from a single image with implicit representation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8833–8842 (2021) 

- [6] Fan, Z., Zhu, L., Li, H., Chen, X., Zhu, S., Tan, P.: Floorplancad: A large-scale cad drawing dataset for panoptic symbol spotting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10128–10137 (2021) 

- [7] David, J., Leit˜ao, A.: Getting a handle on floor plan analysis – door classification in floor plans and a survey on existing datasets. Architecture and Planning Journal (APJ) **28** (3), 6 (2023) 

- [8] National Fire Protection Association: NFPA 101: Life Safety Code. (2021). https://www.nfpa.org/codes-and-standards/all-codes-and-standards/ list-of-codes-and-standards/detail?code=101 

- [9] Feng, D., Zhuang, X., Chen, Z., Zhong, S., Qi, Y., Chen, H., Ma, H.: Position information encoding fpn for small object detection in aerial images. Neural Computing and Applications **36** (26), 16023–16035 (2024) 

21 

- [10] Wang, L., Miao, Z., Liu, E.: Uav remote sensing detection and target recognition based on scp-yolo. Neural Computing and Applications **36** (28), 17495–17510 (2024) 

- [11] Xia, C., Li, X., Gao, X., Ge, B., Li, K.-C., Fang, X., Zhang, Y., Yang, K.: Pcdrdff: multi-modal 3d object detection based on point cloud diversity representation and dual feature fusion. Neural Computing and Applications **36** (16), 9329–9346 (2024) 

- [12] Zhu, L., Chen, Z., Wang, B., Tian, G., Ji, L.: Sfss-net: shape-awared filter and sematic-ranked sampler for voxel-based 3d object detection. Neural Computing and Applications **35** (18), 13417–13431 (2023) 

- [13] Wang, R., Yang, J., Xu, Y., Li, H.: A coarse-to-fine small object detection framework based on a background complexity classification strategy. Neural Computing and Applications **36** (19), 11241–11255 (2024) 

- [14] Chai, E., Chen, L., Hao, X., Zhou, W.: Mitigate the scale imbalance via multiscale information interaction in small object detection. Neural Computing and Applications **36** (4), 1699–1712 (2024) 

- [15] Wei, W., Cheng, Y., He, J., Zhu, X.: A review of small object detection based on deep learning. Neural Computing and Applications **36** (12), 6283–6303 (2024) 

- [16] Yuan, J., Xiao, L., Wattanachote, K., Xu, Q., Luo, X., Gong, Y.: Fgnet: Fixation guidance network for salient object detection. Neural Computing and Applications **36** (2), 569–584 (2024) 

- [17] Xu, K., Guo, J.: A multi-source feature extraction network for salient object detection. Neural Computing and Applications **35** (35), 24727–24742 (2023) 

- [18] Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024) 

- [19] Zhang, L., Le, B., Akhtar, N., Lam, S.-K., Ngo, T.: Large Language Models for Computer-Aided Design: A Survey (2025). https://arxiv.org/abs/2505.08137 

- [20] Delalandre, M., Valveny, E., Pridmore, T., Karatzas, D.: Generation of synthetic documents for performance evaluation of symbol recognition & spotting systems. International Journal on Document Analysis and Recognition (IJDAR) **13** (3), 187–207 (2010) 

- [21] Rusi˜nol, M., Borr`as, A., Llad´os, J.: Relational indexing of vectorial primitives for symbol spotting in line-drawing images. Pattern Recognition Letters **31** (3), 188–201 (2010) 

22 

- [22] Informatics, N.I.: LIFULL HOME’S Dataset. Informatics Research Data Repository (2015). https://www.nii.ac.jp/dsc/idr/en/lifull/ 

- [23] Heras, L.-P., Terrades, O.R., Robles, S., S´anchez, G.: Cvc-fp and sgt: a new database for structural floor plan analysis and its groundtruthing tool. International Journal on Document Analysis and Recognition (IJDAR) **18** , 15–30 (2015) 

- [24] Liu, C., Furukawa, Y., Hoiem, D.: Rent3d: Floor-plan priors for monocular layout estimation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10920–10928 (2018) 

- [25] Chu, H., Wang, S., Urtasun, R., Fidler, S.: Housecraft: Building houses from rental ads and street views. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VI 14, pp. 500–516 (2016) 

- [26] Dodge, S., Xu, J., Stenger, B.: Parsing floor plan images. In: 2017 Fifteenth IAPR International Conference on Machine Vision Applications (MVA), pp. 358–361 (2017) 

- [27] Liu, C., Wu, J., Kohli, P., Furukawa, Y.: Raster-to-vector: Revisiting floorplan transformation. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2195–2203 (2017) 

- [28] Sharma, D., Gupta, N., Chattopadhyay, C., Mehta, S.: Daniel: A deep architecture for automatic analysis and retrieval of building floor plans. In: 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), vol. 1, pp. 420–425 (2017) 

- [29] Wu, W., Fu, X.-M., Tang, R., Wang, Y., Qi, Y.-H., Liu, L.: Data-driven interior plan generation for residential buildings. ACM Transactions on Graphics (TOG) **38** (6), 1–12 (2019) 

- [30] Goyal, S., Mistry, V., Chattopadhyay, C., Bhatnagar, G.: Bridge: Building plan repository for image description generation, and evaluation. In: 2019 International Conference on Document Analysis and Recognition (ICDAR), pp. 1071–1076 (2019) 

- [31] Liu, Y., Lai, Y., Chen, J., Liang, L., Deng, Q.: Scut-autoalp: A diverse benchmark dataset for automatic architectural layout parsing. IEICE TRANSACTIONS on Information and Systems **103** (12), 2725–2729 (2020) 

- [32] Vidanapathirana, M., Wu, Q., Furukawa, Y., Chang, A.X., Savva, M.: Plan2scene: Converting floorplans to 3d scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10733–10742 (2021) 

23 

- [33] Dong, S., Wang, W., Li, W., Zou, K.: Vectorization of floor plans based on edgegan. Information **12** (5), 206 (2021) 

- [34] Lu, Z., Wang, T., Guo, J., Meng, W., Xiao, J., Zhang, W., Zhang, X.: Datadriven floor plan understanding in rural residential buildings via deep recognition. Information Sciences **567** , 58–74 (2021) 

- [35] Lv, X., Zhao, S., Yu, X., Zhao, B.: Residential floor plan recognition and reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16717–16726 (2021) 

- [36] Kim, H., Kim, S., Yu, K.: Automatic extraction of indoor spatial information from floor plan image: A patch-based deep learning methodology application on large-scale complex buildings. ISPRS International Journal of Geo-Information **10** (12), 828 (2021) 

- [37] Simonsen, C.P., Thiesson, F.M., Philipsen, M.P., Moeslund, T.B.: Generalizing floor plans using graph neural networks. In: 2021 IEEE International Conference on Image Processing (ICIP), pp. 654–658 (2021) 

- [38] Pizarro, P.N., Hitschfeld, N., Sipiran, I.: Large-scale multi-unit floor plan dataset for architectural plan analysis and recognition. Automation in Construction **156** , 105132 (2023) 

- [39] Park, H., Gu, H., Hong, S., Choo, S.: Developing a robust training dataset for ai-driven architectural spatial layout generation. Applied Sciences **14** (16), 7095 (2024) 

- [40] Van Engelenburg, C., Mostafavi, F., Kuhn, E., Jeon, Y., Franzen, M., Standfest, M., Gemert, J., Khademi, S.: Msd: A benchmark dataset for floor plan generation of building complexes. In: European Conference on Computer Vision, pp. 60–75 (2024) 

- [41] Luo, R., Liu, Z., Cheng, T., Wang, J., Wang, T., Wei, X., Wang, H., Li, Y., Chai, F., Cheng, F., et al.: Archcad-400k: An open large-scale architectural cad dataset and new baseline for panoptic symbol spotting. arXiv preprint arXiv:2503.22346 (2025) 

- [42] Xu, Z., Jha, N., Mehadi, S., Mandal, M.: Multiscale object detection on complex architectural floor plans. Automation in Construction **165** , 105486 (2024) 

- [43] Shehzadi, T., Hashmi, K.A., Pagani, A., Liwicki, M., Stricker, D., Afzal, M.Z.: Mask-aware semi-supervised object detection in floor plans. Applied Sciences **12** (19), 9398 (2022) 

- [44] He, K., Gkioxari, G., Doll´ar, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2961–2969 (2017) 

24 

- [45] Mishra, S., Hashmi, K.A., Pagani, A., Liwicki, M., Stricker, D., Afzal, M.Z.: Towards robust object detection in floor plan images: A data augmentation approach. Applied Sciences **11** (23), 11174 (2021) 

- [46] Cai, Z., Vasconcelos, N.: Cascade r-cnn: Delving into high quality object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6154–6162 (2018) 

- [47] Jakubik, J., Hemmer, P., V¨ossing, M., Blumenstiel, B., Bartos, A., Mohr, K.: Designing a human-in-the-loop system for object detection in floor plans. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, pp. 12524– 12530 (2022) 

- [48] Bochkovskiy, A., Wang, C.-Y., Liao, H.-Y.M.: Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934 (2020) 

- [49] Park, S., Kim, H.: 3dplannet: generating 3d models from 2d floor plan images using ensemble methods. Electronics **10** (22), 2729 (2021) 

- [50] Surikov, I.Y., Nakhatovich, M.A., Belyaev, S.Y., Savchuk, D.A.: Floor plan recognition and vectorization using combination unet, faster-rcnn, statistical component analysis and ramer-douglas-peucker. In: International Conference on Computing Science, Communication and Security, pp. 16–28 (2020) 

- [51] Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence **39** (6), 1137–1149 (2016) 

- [52] Sch¨onfelder, P., Stebel, F., Andreou, N., K¨onig, M.: Deep learning-based text detection and recognition on architectural floor plans. Automation in Construction **157** , 105156 (2024) 

- [53] Jocher, G., Chaurasia, A., Stoken, A., Borovec, J., NanoCode012, ChristopherSTAN, Laughing, tkianai, yxNONG, Skalski, P., Imyhxy, Tsai, W., Wang, Z., Borovec, J., Labonte, D., Hogan, A., Chiu, C.-Y., Bochkovskiy, A.: YOLOv5. Zenodo (2022). https://doi.org/10.5281/zenodo.7347926 . https://doi.org/10.5281/ zenodo.7347926 

- [54] Wang, C.-Y., Bochkovskiy, A., Liao, H.-Y.M.: Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7464– 7475 (2023) 

- [55] Jocher, G., Chaurasia, A., Qiu, J.: YOLOv8. https://github.com/ultralytics/ ultralytics (2023) 

- [56] Wang, C.-Y., Yeh, I.-H., Liao, H.-Y.M.: You only learn one representation: Unified 

25 

network for multiple tasks. Journal of Information Science and Engineering **39** (3), 691–709 (2023) 

- [57] Oh, J., Hong, S., Choi, B., Ham, Y., Kim, H.: Integrating text parsing and object detection for automated monitoring of finishing works in construction projects. Automation in Construction **174** , 106139 (2025) 

- [58] Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767 (2018) 

- [59] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755 (2014) 

- [60] Zong, Z., Song, G., Liu, Y.: Detrs with collaborative hybrid assignments training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6748–6758 (2023) 

- [61] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: European Conference on Computer Vision, pp. 213–229 (2020). Springer 

- [62] Zentgraf, S., K¨onig, M.: Enhancing information extraction from building standards with chatgpt-4: A multimodal approach. In: International Conference on Computing in Civil and Building Engineering, pp. 234–247 (2024). Springer 

- [63] OpenAI: Introducing GPT-4.1 in the API (2025). https://openai.com/index/ gpt-4-1/ 

- [64] Anonymous: Door Object Detection Dataset. Roboflow (2024) 

- [65] Anonymous: Floor Plans 500 Dataset. Roboflow (2022) 

- [66] Everingham, M., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The pascal visual object classes (voc) challenge. International journal of computer vision **88** (2), 303–338 (2010) 

- [67] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., _et al._ : An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2020) 

- [68] Wang, W., Dai, J., Chen, Z., Huang, Z., Li, Z., Zhu, X., Hu, X., Lu, T., Lu, L., Li, H., _et al._ : Internimage: Exploring large-scale vision foundation models with deformable convolutions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14408–14419 (2023) 

26 

- [69] Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L., Shum, H.-Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In: The Eleventh International Conference on Learning Representations 

- [70] Ren, T., Yang, J., Liu, S., Zeng, A., Li, F., Zhang, H., Li, H., Zeng, Z., Zhang, L.: A strong and reproducible object detector with only public datasets. arXiv preprint arXiv:2304.13027 (2023) 

- [71] Yang, J., Li, C., Dai, X., Gao, J.: Focal Modulation Networks (2022) 

- [72] Liu, S., Ren, T., Chen, J., Zeng, Z., Zhang, H., Li, F., Li, H., Huang, J., Su, H., Zhu, J., _et al._ : Detection transformer with stable matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6491–6500 (2023) 

- [73] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255 (2009). Ieee 

- [74] Wightman, R.: PyTorch Image Models. GitHub (2019). https://doi.org/10.5281/ zenodo.4414861 

- [75] Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., Cao, Y.: Eva: Exploring the limits of masked visual representation learning at scale. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19358–19369 (2023) 

- [76] Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: Scaling vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12104–12113 (2022) 

27 

