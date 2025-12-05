# MoleQ-M-L: Quantum vs Classical ML for Molecules

Comparing quantum machine learning with classical neural networks for predicting molecular energies and forces. Spoiler: quantum's pretty good with symmetry, but classicals hold their own.

---

## What's This?

We're comparing 4 different ML models on predicting molecular properties:

- **LiH** (2 atoms, simple)
- **NH₃** (4 atoms, slightly messier)

The 4 models:
1. Quantum + rotational symmetry (quantum's best shot)
2. Quantum baseline (no symmetry tricks)
3. Quantum + graph structure (for when you have more atoms)
4. Classical NN + symmetry (the practical baseline)

All of them predict molecular energies and forces pretty well. Using symmetry helps a lot. That's the main takeaway.

---

## Get Up and Running

### 1. Install

```bash
git clone https://github.com/sbisw002/MoleQ-M-L.git
cd MoleQ-M-L

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run It

```bash
jupyter notebook "Run LiH comp4.ipynb"
```

Just run all cells. It'll:
- Load the data (already in the repo)
- Train all 4 models
- Dump results in `lih_results/`

### 3. Look at Results

```bash
jupyter notebook "Reader LiH.ipynb"
```

Run all cells, get comparison charts. Done.

---

## What You Can Actually Do

### Path 1: Quick Comparison (Most People)
```
Run LiH comp4.ipynb → Reader LiH.ipynb
Run NH3 comp4.ipynb → Reader NH3.ipynb
```
**Output:** Charts showing which model's best

### Path 2: Generate Fresh Data (If You're Into That)
```
DFT_psi4_LiH.ipynb → new molecular data
```
Uses DFT calculations to generate training data.  
**Need:** Psi4 installed

### Path 3: Do Statistical Testing (Overkill For Most)
```
Run k fold comparison LiH.ipynb → Read k fold results LiH.ipynb
```
Tests if the differences between models are actually significant.  

---

## Folder Layout

```
MoleQ-M-L/
├── README.md                         ← you are here
├── TECHNICAL_DETAILS.md
├── requirements.txt
│
├── Run LiH comp4.ipynb               ← train & compare LiH
├── Run NH3 comp4.ipynb               ← train & compare NH₃
├── Reader LiH.ipynb                  ← visualize LiH results
├── Reader NH3.ipynb                  ← visualize NH₃ results
├── DFT_psi4_LiH.ipynb
├── DFT_psi4_NH3.ipynb
├── Run k fold comparison LiH.ipynb
├── Read k fold results LiH.ipynb
│
├── eqnn_force_field_data_LiH/        ← pre-generated LiH data
├── eqnn_force_field_data_nh3_new/    ← pre-generated NH₃ data
│
├── lih_results/                      ← created when you run experiments
├── nh3_results/
├── kfold_results_lih/
├── kfold_results/
└── figures/
```

---

## Messing With It

### Want to change the experiment?
1. Open `Run LiH comp4.ipynb`
2. Tweak the parameters at the top:
   - `n_runs=3` — how many times to run training
   - `n_epochs=200` — training loops
3. Results go to `lih_results/`

### Getting errors?
See troubleshooting below.

---

## When Things Break

**JAX can't find my GPU?**
```python
import jax
print(jax.devices())  # should show GPU
```

If it's empty:
```bash
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Out of memory?**
Shrink the model in the notebook:
```python
RotationallyEquivariantQML(n_qubits=4, depth=4)  # smaller than 6, 6
```

**Psi4 is a pain to install?**
Skip it. The datasets are already in the repo. Only need Psi4 if generating fresh data.

**Loss goes NaN during training?**
Reduce the learning rate in the notebook (divide by 10).

---

## Dependencies

- **pennylane** — quantum ML framework
- **jax** — autodiff magic
- **numpy, scipy** — math stuff
- **matplotlib** — plotting
- **scikit-learn** — preprocessing
- **psi4** — DFT calculations (optional, only for new data)

Full list: `requirements.txt`

---

## Cite This

```bibtex
@software{moleq_ml_2025,
  title = {MoleQ-M-L: Quantum Machine Learning for Molecular Property Prediction},
  year = {2025},
  url = {https://github.com/sbisw002/MoleQ-M-L},
  author = {Saumya Biswas and Jiten Oswal},
  note = {Quantum machine learning framework comparing equivariant quantum circuits with classical neural networks for molecular energy and force prediction},
  keywords = {machine learning, neural networks, molecular prediction, equivariance, quantum circuits, quantum computing},
  abstract = {Comparative study of quantum machine learning versus classical neural networks for predicting molecular energies and forces, emphasizing the importance of symmetry-aware models in quantum and classical approaches}
}
```

---

## Questions or Issues?

Open an issue on GitHub or create a PR (from your fork) if you want to contribute to this project. 

---

**Last Updated:** December 2025  
**Version:** 0.8.4
