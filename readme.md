
# Layer based system for dimensionality reduction

![alt text](examples/example_plot.png)
TODO: find a name and give a brief explanation

## Setup

TODO:
explain setup

## Steps

1. Clean the data with preprocess/clean.py to get rid of empty datapoints
2. Run one of the scripts in embedding to save the documents as embeddings, there are 3 options for models (bert, gemma300m, vaultgemma) provided
3. Run the dimensionality reduction gridsearch. example: `gridsearch.py -f embeddingdata.npz -r 1 -t PCA -s 0 -t2 LDA`
    - -f, --embedding-file Path to the embedding file
    - -r, --runs Number of runs to execute
    - --run used to run a specific run (for HPC systems or redoing specific runs)
    - -t, --dr-type The primary type of dimensionality reduction (DR) that happens in the first layer
    - t2 --dr-type-secondary The secondary type of DR that occurs after noise is added (default is the same as primary)
    - -s --resolution Changes the grid size resolution in a range of 0-4
    - --no-plot Removes the automatic plotting from the gridsearch script
4. Plot results with: `python plot_results.py -c runs/` 
    - Option 1: -c --crawl Crawls through a directory recursively to find all output files
    - Option 2: -d --directory Choose one directory to make into plots
    - Option 3: -f --files Choose files to add manually
    - -o --output Change output file name
    - -ao --average-ouput Change the average plot's file name
5. Compare graphs and find the best balance for the required epsilon
6. Make/edit a json layer file (examples in examples/) then run `python Factory.py path_to_json`
    - The layers are executed from top to bottom and there can be as many layers as needed.

## Dimensionality Reduction types:

    - PCA 
    - TSNE
    - LDA
    - SVD
    - MDS
    - LLE
    - SOM
    - UMAP

## Metrics

    - Contiunity
    - Trustworthiness
    - Cluster Ordering
    - Pearson Correlation
    - Spearman Correlation
    - Silhouette Score
    - Average of the above metrics
    - CPU Process time
    - Estimated number of clusters
    

## Resolutions: Epsilons | output dimensions

    - 0 [1,2] | [768,2]
    - 1 [1,10,50,100,500,1000] | [768,3,2]
    - 2 [1,10,50,100,500,1000] | [768,384,128,48,8,2]
    - 3 [1,10,50,100,500,1000,5000,10000] | [768,512,256,128,64,32,16,8,4,2]
    - 4 [.1,.5,1,5,10,25,50,100,250,500,1000,2500,5000,10000] | [768,512,256,128,96,64,32,16,12,8,6,4,3,2]
