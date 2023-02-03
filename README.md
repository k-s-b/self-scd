# Statistical Climate Downscaling sans Ground Truth Data

#### This is the Pytorch implementation "Statistical Climate Downscaling sans Ground Truth Data".

# Abstract

Climate change is one of the most critical challenges that our planet is facing today. Rising global temperatures are already bringing noticeable changes to Earth's weather and climate patterns with increased frequency of unpredictable and extreme weather events. Future projections for climate change research are largely based on Earth System Models (ESM), which are computer models that simulate the Earth's climate system as a whole. ESM models are a framework to integrate various physical systems and their output is bound by the enormous computational resources required for running and archiving higher-resolution simulations. In a given budget of the resources, the ESM are generally run on a coarser grid, followed by a computationally lighter downscaling process to obtain a finer resolution output. In this work, we present a deep-learning model for downscaling ESM simulation data that does not require high-resolution ground truth data for model optimization. This is realized by leveraging salient data-distribution patterns and hidden dependencies between the weather variables for an individual data point at runtime. Extensive evaluation on 2x, 3x, and 4x scaling factors demonstrates that the proposed model consistently obtains superior performance over various baselines. Improved downscaling performance and no dependence on high-resolution ground truth data make the proposed method a valuable tool for climate research and mark it as a promising direction for future research.

## Model Architecture
![Screenshot 2023-02-03 204729](https://user-images.githubusercontent.com/62580782/216596023-2af5174b-571d-4bd2-9d1e-614e9180f676.jpg)

## Training flow
![training flow](https://user-images.githubusercontent.com/62580782/216606551-ab89b043-99f0-4a81-926b-63db8dca26ad.jpg)


## Usage
To run model change data paths for the following variables `largeScaleDataset`, `largeScaleDataset_test`, `metaTransferDataset`, `dataset` in `main.py`

## Data

High resolution CESM data can be downloaded from Ultra-high-resolution climate simulation project webiste [at this link](http://climatedata.ibs.re.kr/data/cesm-hires)
