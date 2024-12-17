import random
import pandas as pd


def general_negative_comments(df, general_negative_comments=None):
    # Group by user to get a list of categories they interacted with
    user_categories = df.groupby('usr_id')['category'].apply(set).to_dict()

    # Get a mapping of all articles by category
    category_articles = df.groupby('category')['article_id'].apply(list).to_dict()
    
    if general_negative_comments is None:
        general_negative_comments = [
                # Not one's cup of tea
                "Bài viết này không liên quan đến sở thích của tôi.",
                "Tôi không quan tâm đến chủ đề này.",
                "Chủ đề này không thực sự hấp dẫn tôi.",
                
                # Not understand content
                "Tôi không hiểu ý nghĩa của bài viết này.",
                "Nội dung này có vẻ phức tạp và khó tiếp cận.",
                "Tôi thấy bài viết này không rõ ràng.",
                
                # Not like the category
                "Tôi không thích đọc các bài viết thuộc thể loại này.",
                "Thể loại này không thu hút tôi lắm.",
                "Tôi thường tránh đọc bài về chủ đề này.",
                
                # Not relevant
                "Bài viết này không liên quan gì đến tôi.",
                "Tôi không thấy mình có mối quan tâm gì với nội dung này.",
                "Chủ đề này không phù hợp với tình hình của tôi.",
                
                # Too popular or boring
                "Những thông tin này quá nhàm chán và không có gì mới.",
                "Tôi đã đọc nhiều bài tương tự rồi.",
                "Bài viết không có điểm nào nổi bật để giữ tôi đọc."
        ]

    all_categories = set(df['category'].unique())

    negative_samples = []

    for usr_id, user_cats in user_categories.items():
        # Get the user's positive interactions
        positive_articles = df[df['usr_id'] == usr_id]['article_id'].tolist()
        num_positives = len(positive_articles)
        
        # Determine the number of negatives to sample (e.g., 2:1 ratio)
        num_negatives = num_positives  # Adjust ratio as needed
        
        # Categories the user hasn't interacted with
        negative_categories = all_categories - user_cats
        negative_candidates = [
            article for cat in negative_categories for article in category_articles.get(cat, [])
        ]
        
        # Sample negatives proportional to the user's positives
        sampled_negatives = random.sample(negative_candidates, min(num_negatives, len(negative_candidates)))
        
        # Create negative samples
        for article_id in sampled_negatives:
            template = df.loc[df.article_id == article_id].sample(1).iloc[0].to_dict()
            template['usr_id'] = usr_id
            template['label'] = 0
            template['user_comment'] = random.sample(general_negative_comments, k=1)[0]
            negative_samples.append(template)
            # negative_samples.append({
            #     'usr_id': usr_id,
            #     'article_id': article_id,
            #     'category': train.loc[train['article_id'] == article_id, 'category'].values[0],
            #     'label': 0  # Negative label
            # })
    nega_df = pd.DataFrame(negative_samples)
    new_df = pd.concat([df, nega_df])
    return new_df
