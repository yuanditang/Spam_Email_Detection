# Malicious Email Detection via Browser-Based Extension

### Team Members  
**Alex Koehler, Feng-Jen Hsieh, Yuandi Tang, Vy Le**

---

## Overview

Email remains one of the most exploited vectors for cybercrime. Traditional spam filters—like those in Gmail—struggle to detect advanced phishing, Business Email Compromise (BEC), and socially engineered attacks.  

This project introduces a **browser-based email security extension** powered by a **hierarchical ensemble of specialized models**. Instead of flattening diverse threat datasets into a single classifier, we design **four specialized models** for different malicious content types (phishing, advance-fee fraud, malware, and BEC), orchestrated by a **meta-classifier**.  

We further integrate **explainable AI (XAI)** techniques—attention visualization, SHAP value analysis, and rule extraction—to help users understand why an email is classified as malicious. The ultimate goal: **improve both detection accuracy and user trust through transparency**.

---

## Key Research Questions

1. Do **type-specific models** outperform generic “catch-all” classifiers for email security?  
2. Can we extract **interpretable decision rules** from complex ML models that remain actionable to users?  
3. Do **explanatory interfaces** actually improve human decision-making when evaluating potential threats?

---

## Datasets

We leverage **seven complementary datasets** representing diverse threat types and eras of email attacks:

| Dataset | Focus | Notable Features |
|----------|--------|------------------|
| **Nazario Phishing Corpus** | Credential phishing | Classic phishing patterns (2004–2015) |
| **Nigerian Fraud Collections** | Advance-fee scams | Distinct narrative & linguistic patterns |
| **TREC Spam Track (05–07)** | Commercial spam | Real-world inbox data, nuanced labels |
| **EPVME** | Visual phishing | HTML/CSS obfuscation & image-based deception |
| **Bitabuse** | Crypto scams & ransomware | Emerging blockchain-related fraud |
| **Recent Phish Feeds (PhishTank, OpenPhish)** | Modern phishing (2023–2024) | AI-generated & COVID-themed scams |
| **Legitimate Business Emails (Enron + others)** | Non-malicious BEC-like emails | Executive tone, transaction discussions |

Each dataset is harmonized into a **hierarchical label taxonomy**, preserving distinctions like fraud subtypes and target brands.

---

## Architecture

Our **three-tier hierarchical ensemble** integrates efficiency, specialization, and interpretability:

1. **Tier 1 – Gating Classifier (Logistic Regression)**  
   - Performs quick triage (legitimate / malicious / uncertain).  
   - Lightweight, interpretable, and computationally efficient.  

2. **Tier 2 – Specialized Classifiers**  
   - **Phishing Detector:** DistilBERT fine-tuned on Nazario, EPVME, modern phishing.  
   - **Advance-Fee Fraud Detector:** XGBoost with TF-IDF on Nigerian fraud corpora.  
   - **Malware Detector:** Random Forest using structural + attachment features.  
   - **BEC Detector:** DistilBERT with contextual anomaly features (sender-recipient patterns, urgency language).  

3. **Tier 3 – Meta-Classifier (Neural Attention Network)**  
   - Aggregates specialist outputs, weighting them via attention mechanisms.  
   - Produces a unified confidence score and interpretability layer.

---

## Explainability & User Interface

Our browser extension provides **multi-level interpretability**:

| Technique | Purpose |
|------------|----------|
| **Attention Visualization** | Highlights key text phrases influencing classification |
| **SHAP Values** | Quantifies feature importance using game-theoretic contribution scores |
| **Rule Extraction (inTrees)** | Converts tree ensembles into human-readable IF–THEN rules |
| **Natural Language Explanations** | Converts technical output into plain-language insights |

### Browser Extension UI
- Simple threat badges (🟢 Safe / 🟡 Suspicious / 🔴 Dangerous)  
- Expandable “Why?” section showing primary indicators  
- Optional advanced mode for technical users (attention maps, SHAP charts, rules)

---

## Preliminary Results

| Model | Dataset | F1 Score | Notes |
|--------|----------|----------|-------|
| Logistic Regression (baseline) | Nazario + TREC | 85.9% | Strong linear baseline |
| Unified DistilBERT | Mixed Phishing + Fraud | 89.2% | High phishing accuracy, low fraud recall |
| Specialist DistilBERT (Phishing) | Phishing only | **92.8%** | Focused semantic performance |
| Specialist XGBoost (Fraud) | Fraud only | **88.6%** | +7.2% improvement over unified model |

🧩 **Key Insight:** Specialist models outperform unified classifiers, particularly on minority threat types.

---

## Research Pipeline

**1. Data Processing**
- Format normalization (MIME, HTML, text)
- Near-duplicate detection using MinHash LSH  
- Hierarchical label taxonomy construction  
- Class balancing via stratified sampling + text-based SMOTE  

**2. Exploratory Analysis**
- Vocabulary overlap, cross-dataset validation  
- Semantic clustering (K-means, DBSCAN on TF-IDF & DistilBERT embeddings)  
- UMAP visualizations for dataset separability  

**3. Modeling**
- Tiered ensemble training with cross-validation  
- Temporal holdout testing (2024 data)  
- Adversarial robustness tests (character obfuscation, paraphrasing)

**4. Interpretation & Evaluation**
- SHAP & attention-based interpretability reports  
- User studies measuring explanation effectiveness  
- Expert validation of interpretability correctness  

---

## System Implementation

| Component | Technology Stack |
|------------|------------------|
| **ML Backend** | Python, PyTorch, scikit-learn, HuggingFace Transformers |
| **API Layer** | Flask REST API, asynchronous task queue |
| **Frontend** | Chrome/Firefox extension (JavaScript + Manifest V3) |
| **Deployment** | Cloud GPU instance (for DistilBERT inference), Dockerized services |

---

## Project Timeline

| Phase | Weeks | Key Deliverables |
|--------|--------|------------------|
| Data acquisition & preprocessing | 1–2 | Unified corpus + taxonomy |
| Modeling (gating + specialists) | 3–5 | Trained & validated models |
| Interpretation system | 6 | Attention/SHAP/rule modules |
| Backend API development | 7–8 | RESTful model serving |
| Browser extension & integration | 9–10 | Working prototype & user testing |

---

## Risk Mitigation

- **Evolving threats:** Use 2023–2024 validation emails, anomaly detection, and incremental retraining.  
- **Compute constraints:** Cloud GPUs, model distillation, and quantization.  
- **Cross-platform extension issues:** Modular design, early prototyping, Gmail-first focus.  
- **False positives:** Hierarchical gating + conservative precision tuning.  

---

## Success Metrics

| Evaluation Aspect | Goal |
|--------------------|------|
| **Detection** | ≥85% precision / ≥82% recall overall |
| **Generalization** | ≥80% F1 retention on 2024 holdout data |
| **Adversarial Robustness** | Detect ≥70% of crafted evasions |
| **Latency** | <200ms (gating), <2s (full analysis) |
| **User Study Results** | +15% accuracy improvement with explanations |

---

## Research Contributions

- **Hierarchical multi-specialist ensemble** for email threat detection  
- **Multi-dataset harmonization pipeline** spanning 20 years of threat evolution  
- **Explainable AI integration** for cybersecurity decision support  
- **Browser-based deployment** bridging ML and end-user usability  
- **Empirical evaluation** of interpretability effectiveness  

---

## Citation

If referencing this project in research or documentation:

```bibtex
@misc{malicious_email_extension2025,
  title={Malicious Email Detection via Browser-Based Extension},
  author={Vy Le and Alex Koehler and Feng-Jen Hsieh and Yuandi Tang},
  year={2025},
  note={Graduate Research Project, Northeastern University}
}
```
---

## Future Work

- **Real-time phishing feed integration:** Incorporate live threat feeds for continual model updates and adaptation to emerging attack patterns.  
- **Expanded platform support:** Extend browser extension compatibility to Outlook, Yahoo Mail, and enterprise email clients.  
- **LLM-based anomaly detection:** Explore large language models for detecting subtle, context-dependent threats beyond keyword or pattern matching.  
- **User personalization:** Tailor threat sensitivity and explanation granularity based on individual user preferences and risk tolerance.  
- **Continuous interpretability research:** Evaluate effectiveness of new explanation techniques to further improve user decision-making and trust.  
- **Automated BEC detection refinement:** Collect additional BEC examples and refine models to better capture subtle business communication anomalies.  
