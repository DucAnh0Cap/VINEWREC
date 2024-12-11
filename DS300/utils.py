def get_news(news, users):
    news_id = news.article_id.to_list()
    news_lst = []
    for id in news_id:
        news_dict = {}
        news_dict = news.loc[news['article_id'] == id].iloc[0].to_dict()  # Create a copy of news_template
        news_dict['reader'] = users.loc[users['news_id'] == id].Id.to_list()
        news_dict['comments'] = users.loc[users['news_id'] == id].user_comment.to_list()
        news_lst.append(news_dict)
    return news_lst


def get_users(news, users):
    user_ids = users.Id.to_list()
    user_lst = []

    categories = news.category.unique()
    map_dict = {}
    for c_ in categories:
        map_dict[c_] = 0

    for id in user_ids:
        user_dict = dict()
        df_ = users.loc[users['Id'] == id]
        user_dict = df_.iloc[0].drop(['Title', 'news_id', 'user_comment', 'time_com', 'avata_coment_href']).to_dict()
        user_dict['comments'] = df_.user_comment.to_list()
        user_dict['news_id'] = df_.news_id.to_list()
        user_dict['categories'] = []
        for news_id in user_dict['news_id']:
            category = news.loc[news['article_id'] == news_id]
            if category.shape[0] != 0:
                user_dict['categories'].append(category.iloc[0].category.split(',')[-1])

        temp_dict = map_dict.copy()
        for c_ in user_dict['categories']:
            temp_dict[c_] += 1
        user_dict['interactec_rate'] = list(temp_dict.values())
        user_lst.append(user_dict)
    return user_lst
