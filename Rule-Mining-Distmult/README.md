## Rule Mining from Distmult

[Paper](https://arxiv.org/pdf/1412.6575.pdf) Section 5

3 files are required for mining and infering:

**Rule_Mining.{ipynb/.py}**: Mines rules of length 1,2 and 3. Look at python file for all the inputs required. The required inputs are all listed at top with a comment `#INPUT`  
**pruning_rules.{ipynb/.py}**: Prunes rules of length 1 and 2 and makes them in format same as length 3 rules. The required inputs are all listed at top with a comment `#INPUT`  
**Answer_triplets.{ipynb/.py}**: Answers triple based on mined rules. Look at python file for all the inputs required. The required inputs are all listed at top with a comment `#INPUT`  

Apart, for understanding, there is `analyse_distribution.ipynb`. It does some preliminary analysis to find correct support thresholds for your system. 

**Note:** Python files are not tested. There counter-parts jupyter notebooks are tested and used to generate results.
