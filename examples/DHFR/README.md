# Trimethoprim resistance in Pneumocystis Jirovecci

We use a collection dihydrofolate reductase variants from *Pneumocystis Jirovecii*  (DHFR<sub>pj</sub>) that have been characterized biochemicaly by Queener et al.<sup>[1]</sup>  

DHFR is the primary target of the anti-microbial drug trimethoprim and all DHFR variants in this study have been obtained from clinical isolates of *P. Jirovecii*.

In lieu of a crystal structure of DHFR<sub>pj</sub>, homology models of the DHFR<sub>pj</sub> -- TMP complex were generated using the structure of DHFR from *Pneumocystis Carinii* as a template (PDB ID: 1DYR)<sup>[2]</sup> 100ns pMD simulations of the complex were run in triplicates and ROBUST descriptors, from which robust descriptors were calculated.

Because of the large size of the dataset, 105 molecular dynamics simulations and ~2GB of raw data, we only provide the final descriptors, however other data can be made available upon request.

### Notebooks:

* **calculate_descriptors**

    Demonstrates how to calculate ROBUST descriptors from molecular dynamics simulations


* **predict_TMP_resistance**

    Analysis of the DHFR<sub>pj</sub> dataset, concluding with the establishment of a predictive model for TMP resistance in DHFR<sub>pj</sub>.


### References
[1] [Queener SF, Cody V, Pace J, Torkelson P, Gangjee A. Trimethoprim resistance of dihydrofolate reductase variants from clinical isolates of Pneumocystis jirovecii. Antimicrobial agents and chemotherapy. 2013 Oct 1;57(10):4990-8.](https://aac.asm.org/content/57/10/4990.short)

[2] [Champness JN, Achari A, Ballantine SP, Bryant PK, Delves CJ, Stammers DK. The structure of Pneumocystis carinii dihydrofolate reductase to 1.9 Å resolution. Structure. 1994 Oct 1;2(10):915-24.](https://www.sciencedirect.com/science/article/pii/S096921269400093X)
