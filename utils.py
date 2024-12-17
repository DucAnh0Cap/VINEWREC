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
    article_ids = df['article_id'].unique()  # No need to convert to list explicitly
    
    # Drop columns outside the loop for efficiency
    df = df.drop(['author_url', 'author_description', 'content', 'No_Title', 
                  'avata_coment_href', 'time_com', 'nli_score', 'nickname', 
                  'user_reacted', 'publish_date', 'author_name'], axis=1)
    
    article_lst = []
    for id in article_ids:
        article_df = df[df['article_id'] == id]
        article_dict = article_df.iloc[0].to_dict()  
        
        # Using .tolist() directly for efficiency
        article_dict['usr_ids'] = article_df['usr_id'].tolist()  
        article_dict['comments'] = article_df['user_comment'].tolist()
        article_dict['labels'] = article_df['label'].to_list()
        article_dict.pop('usr_id')
        article_dict.pop('label')
        article_lst.append(article_dict)

    return article_lst


def get_users(df, all_data):
    user_lst = []

    # Create a mapping for categories
    # categories = df['category'].unique()
    categories = ['Chính trị & chính sách', 'Covid-19', 
                  'Giáo dục & tri thức', 'Góc nhìn', 
                  'Kinh doanh & quản trị', 'Môi trường', 
                  'Văn hóa & lối sống', 'Y tế & sức khỏe']


    # Filter the DataFrame to include only rows where label == 1
    filtered_df = df[df['label'] == 1]

    # Group by 'usr_id' and aggregate
    grouped = df.groupby('usr_id').agg(
        comments=('user_comment', list),
        articles_id=('article_id', list),
        tags=('tags', list),
        labels=('label', list),
        categories=('category', lambda x: list(set(x))),
    ).reset_index()

    for _, user_data in grouped.iterrows():
        user_dict = user_data.to_dict()
        
        # Get actual interacted_categories (categories that has label = 1)
        liked_cat = filtered_df.loc[filtered_df.usr_id == user_data['usr_id']].category.to_list()
        user_dict['interacted_categories'] = [1 if cat in liked_cat else 0 for cat in categories]
        
        # Create a interacted-category rate list
        temp_dict = {cat: 0 for cat in categories}
        for a_id in user_dict['articles_id']:
            cat_ = df.loc[df['article_id'] == a_id, 'category'].iloc[0]
            temp_dict[cat_] += 1
        
        liked_cat = filtered_df.loc[filtered_df.usr_id == user_data['usr_id']]

        user_dict['interacted_rate'] = list(temp_dict.values())
        user_lst.append(user_dict)

    return user_lst