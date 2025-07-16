# 🔬 TNBC Clinical Trial Knowledge Graph

> A MVP ; interactive knowledge graph built from real clinical trial data related to **Triple-Negative Breast Cancer (TNBC)**. Explore drugs, biomarkers, gene targets, and trial metadata in a beautifully visualized format.

[![Streamlit](https://img.shields.io/badge/🚀%20Launch%20Live%20App%20on%20Streamlit-green?style=for-the-badge)](https://tnbc-kg-app-fpmvauyivvrlcnaaxqnfte.streamlit.app/)

---

## 🧠 Features

- ✅ Real-world TNBC clinical trials from [ClinicalTrials.gov](https://clinicaltrials.gov)
- ✅ Triplet structure: **(head, relation, tail)** format
- ✅ Drug classification: Biotech / Chemical / Other
- ✅ Gene target enrichment + trial type detection
- ✅ Interactive visualization with **PyVis**
- ✅ Scientific yet minimalist design
- ✅ Deployed using **Streamlit Cloud**

---

## 📁 Project Structure

```bash
├── app.py                  # Streamlit frontend app
├── pipeline.py             # Data extraction + triplet generation
├── utils.py                # Helpers: classification, gene mapping, etc.
├── data/
│   └── tnbc_kg_triplets.csv   # Final KG triplet data
├── requirements.txt        # Streamlit dependencies
└── README.md               # You're reading it!
