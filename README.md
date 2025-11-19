# Customer Feedback Prediction & LLM Strategy Generation System

This project is built around two major components:

## 1.  Machine Learning–Based Prediction System
This module uses a **Random Forest Classifier** to predict customer-related outcomes such as:
- churn probability  
- feedback sentiment classification  

### Key Features
- Data preprocessing and feature engineering  
- Train/test split with model evaluation  
- Hyperparameter tuning  
- Feature importance visualization   

---

## 2.NLP & LLM-Based Strategy Generation Pipeline
This part focuses on **processing actual customer feedback** and generating **actionable retention strategies** using a Large Language Model (Qwen 2.5:3B via Ollama).

The NLP pipeline includes:
- Feedback sentiment extraction  
- Topic modeling / clustering  
- Batch-wise strategy generation using LLM  
- Automatic quality scoring (Action, Empathy, Specificity, Vague signal detection)  
- Clean-up system to normalize and enforce 2-sentence outputs  
- Regeneration loop until the strategy meets quality standards  
- Final evaluated dataset with labels: *Effective* / *Needs Review*  

### LLM Workflow Overview
1. Load sentiment + topic datasets  
2. Batch input is built using feedback + sentiment + topics  
3. Send prompt to local LLM through `ollama_generate()`  
4. Model returns structured JSON with strategies  
5. Clean text to exactly two sentences  
6. Score each strategy using rules  
7. Regenerate any that score < 90  
8. Save results + checkpoints for long runs  

---

## 🗂️ Project Structure

