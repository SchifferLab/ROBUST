# ROBUST

----------------------
Add Description


## Requirements

Transformers require [Schrodinger](https://www.schrodinger.com) &GreaterEqual; v.18.1 to process molecular dynamics simulations.  

Example descriptors have been precomputed and do not require Schrodinger.

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

All transformers  can be incorporated into a PLDB pipeline. For details on our in-house analysis pipeline contact [Flo](florian.leidner@umassmed.edu)


## Examples

1. DHFR
Application of pMD and ROBUST to identify Trimethoprim resistant dihydrofolate reductase. Inhibitor data was taken form Queener et al.[1]


## References

[1] Queener SF, Cody V, Pace J, Torkelson P, Gangjee A. Trimethoprim resistance of dihydrofolate reductase variants from clinical isolates of Pneumocystis jirovecii. Antimicrobial agents and chemotherapy. 2013 Oct 1;57(10):4990-8.

## Contact


Florian Leidner:

florian.leidner@umassmed.edu

Celia Schiffer:

celia.schiffer@umassmed.edu
