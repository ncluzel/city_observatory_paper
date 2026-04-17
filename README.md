# Long-term monitoring through a wastewater-based observatory to model urban population dynamics and health indicators

This GitHub repository contains the code needed to reproduce the article of the same name, available here: https://doi.org/10.1016/j.scitotenv.2026.181675. Raw dataset can be accessed here: https://data.mendeley.com/datasets/733pw8jfwb/2 (make sure to access the second version of the dataset if you want to reproduce the viral data analysis, as I forgot to append in the Astrovirus raw quantifications in the first version).

Disclaimer: Due to (I assume) certain issues related to the precision of floats in Python and the conversion of the files used in writing the article into a single shared dataset, it was not possible to reproduce the article exactly without manually adjusting the values of a few samples (we’re talking about fewer than 10 values out of several thousand). Some code files therefore contain lines dedicated to the “pixel-perfect” reproduction of the article’s figures and tables.

The repository is organized as follows:

root/
├── data/
│   ├── open_medic_data/
│       └── prescriptions.txt
├── libs/
│   ├── SCOU_NC_Vanilla_NUTS_manual.py
│   └── cross_correlation_matcher_specific.py
├── src/
│   ├── om_processing
│       ├── chemical_data_analysis.ipynb
│       └── data_processing_om_data.ipynb
│   ├── population_estimation
│       ├── comparaison_crAss_PMMoV_utils.py
│       ├── comparison_crAssphages_PMMoV_pop.ipynb
│       ├── data_processing_conc_data.ipynb
│       ├── data_processing_flow_data.ipynb
│       ├── data_processing_MES_subsampling_PMMoV_crAssphages.ipynb
│       ├── methods_population_estimation_MAV.ipynb
│       ├── methods_population_estimation_SEV.ipynb
│       ├── models_evaluation.ipynb
│       ├── models_generation_utils.py
│       ├── models_generation.ipynb
│       ├── models_selection.ipynb
│       ├── table_2_reproduction.ipynb
│       └── utils.py
│   └── viral_processing
│       ├── data_processing_viral_data.ipynb
│       └── viral_data_analysis.ipynb
├── requirements.txt
└── README.md

The libs folder contains the source code for the denoising model and the cross-correlation algorithm.
The src folder contains the code needed to reproduce the results presented in the article, organized into three distinct sections covering population estimation, viral data analysis, and organic micropollutants data analysis.

The .python-version and requirements.txt files contain the necessary information to reproduce the python environment used to run these codes.
Feel free to reach out to me for any questions regarding this project, you can find my contact here: https://doi.org/10.1016/j.scitotenv.2026.181675.
