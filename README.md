## WEKA Automated eValuation & Explainability (WEAVE)

WEAVE is a tool for automated machine learning (AutoML), empirical evaluation, and eXplainability AI (XAI) for WEKA-based machine learning models. Given a dataset, WEAVE's pipeline performs: 

- _Model selection:_ uses the automated machine learning tool ML-Plan to search and select a machine learning model (pipeline and hyperparameter)
- _Empirical evaluation_: performs repeated cross-validation on the selected model using the jaicore-experimenter library
- _Explainability_: uses SHAP (SHapley Additive exPlanations) to explain the output of the model with global and local explanations

### How do I get started with WEAVE?