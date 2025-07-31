# VINEWREC: A High-Quality Vietnamese Social News Corpus for Content-Based Recommender Systems
The ViNewRec News Dataset was created to support research in news recommendation systems. It comprises 463 Vietnamese news articles and 11,954 corresponding user comments collected from VnExpress, a leading online news platform in Vietnam. The articles and comments were randomly sampled from content published between December 28, 2021, and November 7, 2024. This diverse dataset covers various domains, including economics, science, education, technology, culture, and society, with all content primarily in Vietnamese.

## Installation:
To set up the project, please follow these steps:
```bash
git clone https://github.com/DucAnh0Cap/VINEWREC_Dataset
cd VINEWREC_Dataset
pip install -r requirements.txt
```

## Usage:
To recreate the experiment:
```bash
python train.py --config_file file_path
                --full_data_file file_path
                --train_file file_path
                --val_file file_path
                --test_file file_path
```



