from tqdm.auto import tqdm
from torchmetrics import Accuracy, Recall, Precision
from torchmetrics.classification import MulticlassF1Score
from tqdm.auto import tqdm


def compute_multiclass_metrics(gens, gts, num_classes=8):
    accuracy_fn = Accuracy(task="multiclass",
                           num_clases=num_classes)
    recall_fn = Recall(task="multiclass",
                       average='micro',
                       num_classes=num_classes)
    precision_fn = Precision(task="multiclass",
                             average='micro',
                             num_classes=num_classes)
    f1_fn = MulticlassF1Score(num_classes=num_classes,
                              average=None)
    accuracy = accuracy_fn(gens, gts)
    recall = recall_fn(gens, gts)
    precision = precision_fn(gens, gts)
    f1 = f1_fn(gens, gts)
    
    return {
        'ACCURACY': accuracy,
        'RECALL': recall,
        'PRECISION': precision,
        'F1': f1,
    }


def get_articles(df):
    article_ids = df.article_id.to_list()
    article_lst = []
    for _, id in tqdm(enumerate(article_ids)):
        article_dict = {}
        article_dict = df.loc[df['article_id'] == id].iloc[0].to_dict()  # Create a copy of news_template
        article_dict['reader'] = df.loc[df['article_id'] == id].usr_id.to_list()
        article_dict['comments'] = df.loc[df['article_id'] == id].user_comment.to_list()
        article_lst.append(article_dict)
    return article_lst


def get_users(df):
    user_ids = df.usr_id.to_list()
    user_lst = []

    categories = df.category.unique()
    map_dict = {'Chính trị & chính sách': 0,
                'Văn hóa & lối sống': 0,
                'Kinh doanh & quản trị': 0,
                'Giáo dục & tri thức': 0,
                'Y tế & sức khỏe': 0,
                'Môi trường': 0,
                'Góc nhìn': 0,
                'Covid-19': 0}
    # for c_ in categories:
    #     map_dict[c_] = 0

    for _, id in tqdm(enumerate(user_ids)):
        user_dict = dict()
        df_ = df.loc[df['usr_id'] == id]

        user_dict = df_.iloc[0].drop(['Title', 'article_id', 'user_comment', 'time_com', 'avata_coment_href', 'content', 'label', 'nli_score', 'tags',
                                      'publish_date', 'No_Title', 'category', 'url', 'author_description', 'author_name', 'author_url']).to_dict()
        user_dict['comments'] = df_.user_comment.to_list()
        user_dict['articles_id'] = df_.article_id.to_list()
        user_dict['categories'] = set()
        user_dict['tags'] = df_.tags.to_list()
        user_dict['nli_scores'] = df_.nli_score.to_list()

        # Get categories

        for art_id in user_dict['articles_id']:
            art_ = df.loc[df['article_id'] == art_id]
            if art_.shape[0] != 0:
                user_dict['categories'].add(art_.iloc[0].category)
        user_dict['categories'] = list(user_dict['categories'])

        # Create a interacted-category list
        temp_dict = map_dict.copy()
        for c_ in user_dict['categories']:
            temp_dict[c_] += 1
        user_dict['interacted_categories'] = list(temp_dict.values()) # A list of 0 and 1 represent the interacted history
        
        # Create a interacted-category rate list
        temp_dict = map_dict.copy()
        for a_id in user_dict['articles_id']:
            cat_ = df.loc[df.article_id == a_id].category.iloc[0]
            temp_dict[cat_] += 1
        user_dict['categories'] = list(categories)
        user_dict['interacted_rate'] = list(temp_dict.values())

        user_lst.append(user_dict)
    return user_lst