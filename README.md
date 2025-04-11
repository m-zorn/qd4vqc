# Quality Diversity for Variational Quantum Circuit Optimization 

Implementation of the quality diversity (QD) approach in [1] to optimizing variational quantum circuits (VQC) with covariance matrix adaptation (CMA). The QD optimization part is based on the `pyribs` library [2], the solutions of which (after conversion) are simulated as `pennylane` [3] quantum circuits.

The generated erdos-reny graphs with `8, 12, 14, 16` vertices can be found in the `data/` directory as pickle files for replication.


### Usage:
```
    # Make virtual env with:
    conda create -n qd4vqc python=3.10
    conda activate qd4vqc
    pip install -r requirements.txt

    # Run the quality diversity experiments (with TINY gateset)
    python src/runnables/qd_exeriments.py
``` 

### References:
[1] Zorn et al., Quality Diversity for Variational Quantum Circuit Optimization, ICAPS 2025 (To appear)

[2] Pyribs QD-optimization (pyribs.org/)

[3] Pennylane quantum simulator (pennylane.ai/)