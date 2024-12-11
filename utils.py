from tqdm import tqdm


def get_articles(articles, users):
    article_ids = articles.article_id.to_list()
    article_lst = []
    for _, id in tqdm(enumerate(article_ids)):
        article_dict = {}
        article_dict = articles.loc[articles['article_id'] == id].iloc[0].to_dict()  # Create a copy of news_template
        article_dict['reader'] = users.loc[users['article_id'] == id].Id.to_list()
        article_dict['comments'] = users.loc[users['article_id'] == id].user_comment.to_list()
        article_lst.append(article_dict)
    return article_lst


def get_users(articles, users):
    user_ids = users.Id.to_list()
    user_lst = []

    categories = articles.category.unique()
    map_dict = {}
    for c_ in categories:
        map_dict[c_] = 0

    for _, id in tqdm(enumerate(user_ids)):
        user_dict = dict()
        df_ = users.loc[users['Id'] == id]

        user_dict = df_.iloc[0].drop(['Title', 'article_id', 'user_comment', 'time_com', 'avata_coment_href']).to_dict()
        user_dict['comments'] = df_.user_comment.to_list()
        user_dict['articles_id'] = df_.article_id.to_list()
        user_dict['categories'] = set()

        for art_id in user_dict['articles_id']:
            art_ = articles.loc[articles['article_id'] == art_id]
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
        for n_id in user_dict['articles_id']:
            cat_ = articles.loc[articles.article_id == 4824889].category.iloc[0]
            temp_dict[cat_] += 1
        user_dict['interacted_rate'] = list(temp_dict.values())
        user_lst.append(user_dict)
    return user_lst
