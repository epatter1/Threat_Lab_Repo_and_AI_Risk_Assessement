# **Weeks 1–2 — Module 1: AI Threat Lab (AWS, Azure, GCP)**


This is Module 1 of 5 of a [12-week Security Architect Program](https://github.com/epatter1/AI_Data_Security_Architect_Program/blob/main/12_week_overview.md)

### Focus areas
* #### Build evasion, extraction, poisoning, inversion demos 
* #### Implement cloud-specific attack surfaces
* #### Create threat models + risk scoring engine
* #### Document findings with diagrams and case studies  
#### Deliverables:

* Threat Lab Repo + AI Risk Assessment Template: [Link to walkthough](https://github.com/epatter1/Threat_Lab_Repo_and_AI_Risk_Assessement/blob/main/Threat_Lab_walkthough.md)
---

## **Technologies, Frameworks & Cloud Services Used**

### 🧠 **AI/ML & Adversarial ML**

<a href="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
</a>
<a href="TensorFlow">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow" />
</a>
<a href="Scikit-learn">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
</a>
<a href="Jupyter">
  <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter" />
</a>
<a href="NumPy">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy" />
</a>
<a href="Pandas">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas" />
</a>
<a href="Matplotlib">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white" alt="Matplotlib" />
</a>

---

### 🔐 **Security, Threat Modeling & Risk**

<a href="MITRE-ATLAS">
  <img src="https://img.shields.io/badge/MITRE-ATLAS-blue" alt="MITRE ATLAS" />
</a>
<a href="OWASP-LLM-Top-10">
  <img src="https://img.shields.io/badge/OWASP-Top_10_for_LLMs-000000?logo=owasp&logoColor=white" alt="OWASP LLM Top 10" />
</a>
<a href="STRIDE">
  <img src="https://img.shields.io/badge/Threat_Modeling-STRIDE-purple" alt="STRIDE" />
</a>
<a href="Risk-Engine">
  <img src="https://img.shields.io/badge/Risk_Scoring-Engine-green" alt="Risk Engine" />
</a>

---

### ☁️ **Cloud Platforms**

### **AWS**

<a href="AWS">
  <img src="https://img.shields.io/badge/AWS-232F3E?logo=amazonaws&logoColor=white" alt="AWS" />
</a>
<a href="SageMaker">
  <img src="https://img.shields.io/badge/Amazon-SageMaker-1F72B8?logo=amazonaws&logoColor=white" alt="SageMaker" />
</a>
<a href="API-Gateway">
  <img src="https://img.shields.io/badge/API_Gateway-FF4F00?logo=amazonaws&logoColor=white" alt="API Gateway" />
</a>
<a href="Lambda">
  <img src="https://img.shields.io/badge/Lambda-FF9900?logo=awslambda&logoColor=white" alt="Lambda" />
</a>
<a href="S3">
  <img src="https://img.shields.io/badge/S3-569A31?logo=amazons3&logoColor=white" alt="S3" />
</a>

---

### **Azure**

<a href="Azure">
  <img src="https://img.shields.io/badge/Azure-0078D4?logo=microsoftazure&logoColor=white" alt="Azure" />
</a>
<a href="Azure-ML">
  <img src="https://img.shields.io/badge/Azure_ML-0078D4?logo=microsoftazure&logoColor=white" alt="Azure ML" />
</a>
<a href="API-Management">
  <img src="https://img.shields.io/badge/API_Management-0089D6?logo=microsoftazure&logoColor=white" alt="API Management" />
</a>
<a href="Azure-Storage">
  <img src="https://img.shields.io/badge/Azure_Storage-0078D4?logo=microsoftazure&logoColor=white" alt="Azure Storage" />
</a>

---

### **GCP**

<a href="GCP">
  <img src="https://img.shields.io/badge/GCP-4285F4?logo=googlecloud&logoColor=white" alt="GCP" />
</a>
<a href="Vertex-AI">
  <img src="https://img.shields.io/badge/Vertex_AI-4285F4?logo=googlecloud&logoColor=white" alt="Vertex AI" />
</a>
<a href="Cloud-Endpoints">
  <img src="https://img.shields.io/badge/Cloud_Endpoints-4285F4?logo=googlecloud&logoColor=white" alt="Cloud Endpoints" />
</a>
<a href="Cloud-Storage">
  <img src="https://img.shields.io/badge/Cloud_Storage-4285F4?logo=googlecloud&logoColor=white" alt="Cloud Storage" />
</a>

---

### 🛠️ **DevOps, Infrastructure & Tooling**

<a href="Python">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
</a>
<a href="Docker">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
</a>
<a href="Terraform">
  <img src="https://img.shields.io/badge/Terraform-844FBA?logo=terraform&logoColor=white" alt="Terraform" />
</a>
<a href="GitHub-Actions">
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white" alt="GitHub Actions" />
</a>

---

### 📊 **Documentation & Visualization**

<a href="Markdown">
  <img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white" alt="Markdown" />
</a>
<a href="Architecture-Diagrams">
  <img src="https://img.shields.io/badge/Architecture_Diagrams-4285F4" alt="Architecture Diagrams" />
</a>
<a href="Case-Studies">
  <img src="https://img.shields.io/badge/Case_Studies-6A5ACD" alt="Case Studies" />
</a>
<a href="Risk-Assessment">
  <img src="https://img.shields.io/badge/Risk_Assessment-Templates-green" alt="Risk Assessment Templates" />
</a>

---

## **WEEK 1 — Foundations + First Attack Demos**

### **1. Set Up the Threat Lab Structure**
```
module1-threat-lab-multicloud/
    notebooks/
    cloud/
        aws/
        azure/
        gcp/
    diagrams/
    docs/
    risk-engine/
```

---

### **2. Build a Baseline Model (Local First)**
Notebook: `notebooks/01_baseline_model.ipynb`

- Train a simple classifier  
- Save model artifacts  
- Document model purpose and threat relevance  

---

### **3. Implement Evasion Attack (FGSM or PGD)**
Notebook: `notebooks/02_evasion_attack.ipynb`

- Generate adversarial examples  
- Compare clean vs adversarial accuracy  
- Visualize perturbations  
- Explain cloud implications (SageMaker, Azure ML, Vertex AI)

---

### **4. Create Initial Threat Model**
Document: `docs/threat_model_week1.md`

Includes:  
- Assets  
- Adversary goals  
- Attack vectors  
- MITRE ATLAS mapping  
- OWASP LLM Top 10 relevance  

---

### **5. Add First Architecture Diagram**
Diagram: `diagrams/week1_architecture.md`

```
┌──────────────────────────────┐
│        Client / Attacker     │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│   Model API (Conceptual)     │
│  AWS / Azure / GCP Endpoint  │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│       Baseline Model         │
│   (Local in Week 1)          │
└──────────────────────────────┘
```

---

## **WEEK 2 — Advanced Attacks + Cloud Attack Surfaces + Risk Engine**

### **1. Implement Model Extraction Attack**
Notebook: `notebooks/03_model_extraction.ipynb`

- Query model repeatedly  
- Reconstruct surrogate model  
- Compare accuracy  
- Document cloud exposure risks  

---

### **2. Implement Data Poisoning Attack**
Notebook: `notebooks/04_data_poisoning.ipynb`

- Inject poisoned samples  
- Retrain model  
- Measure degradation  
- Map poisoning vectors to AWS/Azure/GCP data pipelines  

---

### **3. Implement Model Inversion Attack**
Notebook: `notebooks/05_model_inversion.ipynb`

- Reconstruct input features from outputs  
- Visualize leakage  
- Document privacy implications  

---

### **4. Build the AI Risk Scoring Engine**
Folder: `risk-engine/`

Includes:  
- `risk_engine.py`  
- `risk_factors.yaml`  
- `risk_scoring_example.ipynb`

Outputs:  
- Risk score  
- Recommended controls  
- Cloud exposure level  

---

### **5. Create Cloud-Specific Attack Surface Maps**
Files:  
- `cloud/aws/attack_surface.md`  
- `cloud/azure/attack_surface.md`  
- `cloud/gcp/attack_surface.md`

Each includes:  
- Endpoint exposure  
- IAM risks  
- Data leakage vectors  
- Logging gaps  
- Network isolation weaknesses  

---

### **6. Add Case Studies + Documentation**
Folder: `docs/case_studies/`

Examples:  
- SageMaker extraction attack  
- Azure Data Factory poisoning  
- Vertex AI inversion risk  

---

### **7. Update Architecture Diagrams**
Folder: `diagrams/week2/`

Includes:  
- Multi‑cloud attack surfaces  
- Extraction flow  
- Poisoning flow  
- Inversion flow  

---

### **8. Create the AI Risk Assessment Template**
File: `docs/ai_risk_assessment_template.md`

Sections:  
- Use case  
- Model type  
- Data classification  
- Attack vectors  
- Cloud exposure  
- Risk score  
- Recommended controls  
- Compliance mapping  

---

## **End of Weeks 1–2 Deliverables**

### ✅ **Threat Lab Repository**  
- Baseline model  
- Evasion, extraction, poisoning, inversion demos  
- Cloud attack surface maps  
- Threat models  
- Architecture diagrams  
- Case studies  
- Risk scoring engine  

### ✅ **AI Risk Assessment Template**  
A polished, reusable assessment tool for clients and hiring managers.
