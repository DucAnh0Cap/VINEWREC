# VINEWREC: A High-Quality Vietnamese Social News Corpus for Content-Based Recommender Systems
The ViNewRec News Dataset was created to support research in news recommendation systems. It comprises 463 Vietnamese news articles and 11,954 corresponding user comments collected from VnExpress, a leading online news platform in Vietnam. The articles and comments were randomly sampled from content published between December 28, 2021, and November 7, 2024. This diverse dataset covers various domains, including economics, science, education, technology, culture, and society, with all content primarily in Vietnamese.

## Installation:
To set up the project, please follow these steps:
```bash
git clone https://github.com/DucAnh0Cap/VINEWREC
cd VINEWREC
pip install -r requirements.txt
```

## Usage:
The experiments in this repository are the modified implementation of the following article using our ViNewRec dataset.

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). Neural Collaborative Filtering[http://dl.acm.org/citation.cfm?id=3052569]. In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

To recreate the experiment:
```bash
python train.py --config_file /path/to/your/*.yaml
                --full_data_file /path/to/your/*.csv
                --train_file /path/to/your/*.csv
                --val_file /path/to/your/*.csv
                --test_file /path/to/your/*.csv
```

## Contact
For more information about relevant published papers and datasets:
- Nhi Ngoc Yen Nguyen: 21521231@gm.uit.edu.vn
- Anh Duc Nguyen: 21520140@gm.uit.edu.vn





