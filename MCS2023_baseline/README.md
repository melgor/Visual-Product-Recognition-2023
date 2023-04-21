# Baseline for MCS2023: Visual Product Recognition Challenge
![Visual Products](figures/pipeline.jpg?raw=true "Baseline pipeline")  
This is a repository with a baseline solution for the MCS2023: 
Visual Product Recognition Challenge. 
In this competition, participants need to train a model to search for similar 
products on the marketplaces based on a user's photo.

The idea of the basic solution is to train a classifier of different products, 
remove the classification layer and use embeddings to solve the retrieval problem.

## Steps for working with baseline

### 1. Download Products-10K dataset
The Products-10K dataset is used to train the model. 
You can download it [here](https://products-10k.github.io/).

### 2. Prepare config for training
Prepare `config/baseline_mcs.yml`

### 3. Run model training
```bash                         
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/baseline_mcs.yml
```

<table>
<thead>
<tr>
<th style="text-align:center">query <br/>(user images id)</th>
<th style="text-align:center" colspan=4>gallery <br/>(seller images id)</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">0</td>
<td style="text-align:center">1423</td>
<td style="text-align:center">101</td>
<td style="text-align:center">... (another 997 cells ) ...</td>
<td style="text-align:center">3</td>
</tr>
<tr>
<td style="text-align:center">1</td>
<td style="text-align:center">2744</td>
<td style="text-align:center">56</td>
<td style="text-align:center">... (another 997 cells ) ...</td>
<td style="text-align:center">133</td>
</tr>
<tr>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
<td style="text-align:center">...</td>
</tr>
<tr>
<td style="text-align:center">13999</td>
<td style="text-align:center">4</td>
<td style="text-align:center">199</td>
<td style="text-align:center">... (another 997 cells ) ...</td>
<td style="text-align:center">456</td>
</tr>
</tbody>
</table>
