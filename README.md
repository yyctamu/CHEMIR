# Molecule-Recommendation

## Links
- [Website](https://yaruyang.wixsite.com/mol-rec)
- [Video](https://www.youtube.com/watch?v=R2_QVLiIKYo&t=5s)
- [Summary Report](https://83b66dc9-0e7e-4155-9b49-600d07b0c53e.filesusr.com/ugd/609a2c_1855f7a0acc042929e761885a86e515f.pdf)

## Project Overview
The research Project aims to design and implement a hybrid chemical recommender system. This system will leverage a combination of algorithms, including collaborative filtering, content-based approaches, Graph Neural Networks (GNNs), and autoencoders to effectively identify and recommend compounds of interest. It focuses on introducing scientific researchers to potentially unknown chemical compounds within large-scale chemical datasets, enhancing discovery and research efficiency in chemistry and related fields.

## Methodology
<img width="1028" alt="image" src="https://github.com/PragatiNaikare311/Molecule-Recommendation/assets/143132647/1509cc5d-af12-46d8-b875-a4acc27a437d">


The system leverages users’ past research experiences to recommend innovative synthesized molecules. These recommendations will serve as potential subjects of investigation for researchers in biomedical, drug, and chemistry-related fields. To achieve this, we have developed a powerful molecule recommendation pipeline, tailored to recommend molecules to researchers based on their past interactions. 

## 2-Stage Recommendation Pipeline
Given a researcher’s historical molecule interactions, our recommendation pipeline curates a set of molecules from the MolRec data and generates novel molecules aligned with the researcher’s interests.

### Stage I - Classical Recommendation
#### Collaborative Filtering (CF)

State-of-the-art CF recommender algorithms are employed for implicit data recommendation. Key algorithms utilized include:

- **Alternative Least Squares (ALS)**
- **Bayesian Personalized Ranking (BPR)**

####  Semantic Similarity (Content Based)

Semantic similarity is determined based on the ChEBI ontology, employing the following approach:

- **Chemical Semantic Similarity**: This method calculates similarity between compounds using DiShLn to measure the distance between entities in a semantic base. The similarity metric used here is Resnik.


Results: <img width="1457" alt="Screenshot 2024-05-04 at 11 05 05 AM" src="https://github.com/PragatiNaikare311/Molecule-Recommendation/assets/143132647/ef02b347-f2bb-41ba-bd03-49547d907a43">
<img width="1434" alt="Screenshot 2024-05-04 at 11 04 50 AM" src="https://github.com/PragatiNaikare311/Molecule-Recommendation/assets/143132647/5ef0caef-fd6c-4fa5-89ab-8df2c66b13e3">

ALS_ONTO (Performs Better for Top 5 Molecules) 

### Stage II - Deep Generative Model (VAE-Based Novel Molecule Recommendation)

Researchers' interests may extend beyond molecules present in the MolRec dataset. To address this, we employ a Variational Autoencoder (VAE) to recommend novel molecules not in the MolRec data based on molecules recommended in stage 1.

- **Objective**: Generate novel molecules beyond those in the MolRec dataset.
- **Approach**: Utilize a Variational Autoencoder (VAE) conditioned on molecules recommended in stage 1.
- **JTVAE1**: A 2-stage Graph Neural Network (GNN)-based framework is employed for generating chemically valid molecules.
- **Functionality**: Given an arbitrary number of reference molecules, the VAE generates a new molecule aligned with the researcher's interests.

### Open Source Resources

- **Trained Weights**: We provide open access to the trained weights.
- **Model Demo**: A demonstration of the model is also open-sourced for exploration



<img width="634" alt="image" src="https://github.com/PragatiNaikare311/Molecule-Recommendation/assets/143132647/00234cd4-17b8-4299-8de6-cbe8ba5a11a9">

