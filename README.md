# ROBUST

Molecular dynamics simulations are well suited for studying molecular recognition. A key challenge is the derivation of meaningful descriptors from the raw MD coordinates.
   
ROBUST provides a set of tools to calculate physics based descriptors from molecular dynamics simulations.

Along with the transformers we provide a number of examples cases.

----------------------
Add Description


## Requirements

* python &ge; 3.6
* numpy
* pandas
* scipy
* scikit-learn
* voronota
* [Schrodinger](https://www.schrodinger.com) &ge; v.18.1 

The Schrodinger Software Suite is required to process molecular dynamic simulations.

For examples, the descriptors have been precomputed thus the examples do not require Schrodinger.

## Transformers

A series of transformers to calculate and parse descriptors from molecular dynamics simulations.
Transformers were developed for use with [Schrodingers Protein-Ligand Database](https://www.schrodinger.com/pldb) however, they can also be called from the command line or imported in python.


### Usage

#### From command line

All transformers can be run from the commandline; For a list of arguments simply run:

```
$SCHRODINGER/run python transformer.py --help
```

#### As a python module

All transformers can be imported into a python script or jupyter notebook. An example is shown in the examples/DHFR/calculate_descriptors.ipynb .


#### From the PLDB

All transformers  can be incorporated into a PLDB pipeline. For details on our in-house analysis pipeline contact Judy Huang.


## Examples

#### DHFR

Application of pMD and ROBUST to identify Trimethoprim resistant dihydrofolate reductase variants


## Contact


Florian Leidner:

florian.leidner@umassmed.edu

Judy Huang

qiuyu.huang@umassmed.edu

Celia Schiffer:

celia.schiffer@umassmed.edu


