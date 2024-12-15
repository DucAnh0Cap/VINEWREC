
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# # Step 1: Choose a category
# category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# # Step 2: Display all articles in the selected category
# def get_articles_by_category(category):
#     return df[df['category'] == category]

# category_articles = get_articles_by_category(category_input)

# if not category_articles.empty:
#     # Step 3: User selects a recommendation method
#     recommendation_method = st.radio(
#         "Chọn cách đề xuất bài viết:",
#         ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
#     )

#     # Step 4: Show articles in selected category and allow article selection
#     selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)

#     if selected_article_title:
#         selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

#         st.header(f"{selected_article['title']}")
#         st.write(f"**Ngày**: {selected_article['publish_date']}")
#         st.write(f"**Tác giả**: {selected_article['author_name']}")
#         st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
#         st.write(f"**Từ khoá**: {selected_article['tags']}")
#         st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
#         st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")
        
#         # Step 5: Recommend based on the selected method
#         def recommend_similar_articles(selected_article, category_articles, method):
#             # Create a TF-IDF vectorizer to vectorize the content descriptions
#             vectorizer = TfidfVectorizer(stop_words='english')

#             if method == "Dựa trên bài viết chi tiết":
#                 # Vectorize content descriptions from all articles
#                 descriptions = category_articles['description'].tolist()
#                 descriptions.append(selected_article['description'])  # Add the selected article's description
#                 tfidf_matrix = vectorizer.fit_transform(descriptions)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare with all others
#                 cosine_sim = cosine_sim.flatten()  # Convert to 1D array
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]  # Get top 5 most similar articles
#                 return category_articles.iloc[similar_articles_idx]
            
#             elif method == "Dựa trên từ khóa":
#                 # Vectorize article keywords
#                 keywords = category_articles['tags'].tolist()
#                 keywords.append(selected_article['tags'])  # Add selected article's tags
#                 tfidf_matrix = vectorizer.fit_transform(keywords)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare with all others
#                 cosine_sim = cosine_sim.flatten()
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                 return category_articles.iloc[similar_articles_idx]
            
#             elif method == "Dựa trên tác giả":
#                 # Filter articles by the same author
#                 return category_articles[category_articles['author_name'] == selected_article['author_name']]

#             elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
#                 # Vectorize author descriptions
#                 authors_description = category_articles['author_description'].tolist()
#                 authors_description.append(selected_article['author_description'])
#                 tfidf_matrix = vectorizer.fit_transform(authors_description)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                 cosine_sim = cosine_sim.flatten()
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                 return category_articles.iloc[similar_articles_idx]
        
#         # Get the recommended articles based on the selected method
#         recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_method)
        
#         if not recommended_articles.empty:
#             st.subheader("Các bài viết tương tự:")
#             for _, related_article in recommended_articles.iterrows():
#                 if related_article['title'] != selected_article['title']:  # Avoid showing the same article
#                     st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")
# else:
#     st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")

# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# # Step 1: Choose a category
# category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# # Step 2: Display all articles in the selected category
# def get_articles_by_category(category):
#     return df[df['category'] == category]

# category_articles = get_articles_by_category(category_input)

# if not category_articles.empty:
#     # Step 3: User selects a recommendation method
#     recommendation_method = st.radio(
#         "Chọn cách đề xuất bài viết:",
#         ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
#     )

#     # Step 4: Show articles in selected category and allow article selection
#     selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)

#     if selected_article_title:
#         selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

#         st.header(f"{selected_article['title']}")
#         st.write(f"**Ngày**: {selected_article['publish_date']}")
#         st.write(f"**Tác giả**: {selected_article['author_name']}")
#         st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
#         st.write(f"**Từ khoá**: {selected_article['tags']}")
#         st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
#         st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")
        
#         # Step 5: Recommend based on the selected method
#         def recommend_similar_articles(selected_article, category_articles, method):
#             # Create a TF-IDF vectorizer to vectorize the content descriptions
#             vectorizer = TfidfVectorizer(stop_words='english')

#             if method == "Dựa trên bài viết chi tiết":
#                 # Vectorize content descriptions from all articles
#                 descriptions = category_articles['description'].tolist()
#                 descriptions.append(selected_article['description'])  # Add the selected article's description
#                 tfidf_matrix = vectorizer.fit_transform(descriptions)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare with all others
#                 cosine_sim = cosine_sim.flatten()  # Convert to 1D array
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]  # Get top 5 most similar articles
#                 return category_articles.iloc[similar_articles_idx]
            
#             elif method == "Dựa trên từ khóa":
#                 # Vectorize article keywords
#                 keywords = category_articles['tags'].tolist()
#                 keywords.append(selected_article['tags'])  # Add selected article's tags
#                 tfidf_matrix = vectorizer.fit_transform(keywords)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare with all others
#                 cosine_sim = cosine_sim.flatten()
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                 return category_articles.iloc[similar_articles_idx]
            
#             elif method == "Dựa trên tác giả":
#                 # Filter articles by the same author
#                 return category_articles[category_articles['author_name'] == selected_article['author_name']]

#             elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
#                 # Vectorize author descriptions
#                 authors_description = category_articles['author_description'].tolist()
#                 authors_description.append(selected_article['author_description'])
#                 tfidf_matrix = vectorizer.fit_transform(authors_description)
#                 cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                 cosine_sim = cosine_sim.flatten()
#                 similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                 return category_articles.iloc[similar_articles_idx]
        
#         # Get the recommended articles based on the selected method
#         recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_method)
        
#         if not recommended_articles.empty:
#             st.subheader("Các bài viết tương tự:")
#             for _, related_article in recommended_articles.iterrows():
#                 if related_article['title'] != selected_article['title']:  # Avoid showing the same article
#                     st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")

#         # Step 6: Allow user to continue selecting another article or method
#         continue_button = st.button("Tiếp tục chọn một bài viết khác")
#         if continue_button:
#             st.experimental_rerun()  # Reload the app to allow the user to select again

# else:
#     st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")


# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# # Step 1: Choose a category
# category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# # Step 2: Display all articles in the selected category
# def get_articles_by_category(category):
#     return df[df['category'] == category]

# category_articles = get_articles_by_category(category_input)

# # Step 3: User selects a recommendation method
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Step 4: Add search filter
# search_query = st.text_input("Tìm kiếm bài viết theo tiêu đề hoặc từ khóa:")
# if search_query:
#     category_articles = category_articles[category_articles['title'].str.contains(search_query, case=False, na=False)]

# # Show articles in selected category and allow article selection
# if not category_articles.empty:
#     # Allow multi-method recommendation
#     recommendation_methods = st.multiselect(
#         "Chọn cách đề xuất bài viết:",
#         ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
#     )

#     selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)

#     if selected_article_title:
#         selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

#         st.header(f"{selected_article['title']}")
#         st.write(f"**Ngày**: {selected_article['publish_date']}")
#         st.write(f"**Tác giả**: {selected_article['author_name']}")
#         st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
#         st.write(f"**Từ khoá**: {selected_article['tags']}")
#         st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
#         st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")

#         # Add to history
#         st.session_state.history.append(selected_article['title'])

#         # Step 5: Recommend based on the selected methods
#         def recommend_similar_articles(selected_article, category_articles, methods):
#             vectorizer = TfidfVectorizer(stop_words='english')
#             recommended_articles_all = pd.DataFrame()

#             for method in methods:
#                 if method == "Dựa trên bài viết chi tiết":
#                     descriptions = category_articles['description'].tolist()
#                     descriptions.append(selected_article['description'])
#                     tfidf_matrix = vectorizer.fit_transform(descriptions)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                 elif method == "Dựa trên từ khóa":
#                     keywords = category_articles['tags'].tolist()
#                     keywords.append(selected_article['tags'])
#                     tfidf_matrix = vectorizer.fit_transform(keywords)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                 elif method == "Dựa trên tác giả":
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles[category_articles['author_name'] == selected_article['author_name']]])

#                 elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
#                     authors_description = category_articles['author_description'].tolist()
#                     authors_description.append(selected_article['author_description'])
#                     tfidf_matrix = vectorizer.fit_transform(authors_description)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#             recommended_articles_all = recommended_articles_all.drop_duplicates(subset="title")
#             return recommended_articles_all

#         recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_methods)

#         # Display recommended articles
#         if not recommended_articles.empty:
#             st.subheader("Các bài viết tương tự:")
#             for _, related_article in recommended_articles.iterrows():
#                 if related_article['title'] != selected_article['title']:
#                     st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")

#         # Button to continue after viewing recommendations
#         if st.button("Tiếp tục lựa chọn bài viết khác"):
#             st.rerun()  # This will refresh the app and allow the user to continue choosing

# else:
#     st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")

# # Display history of selected articles
# if st.session_state.history:
#     st.subheader("Lịch sử chọn bài viết:")
#     for entry in st.session_state.history:
#         st.write(f"- {entry}")


# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# # Step 1: Choose a category
# category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# # Step 2: Display all articles in the selected category
# def get_articles_by_category(category):
#     return df[df['category'] == category]

# category_articles = get_articles_by_category(category_input)

# # Step 3: User selects a recommendation method
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Step 4: Add multiselect for filtering articles by keywords
# selected_keywords = st.multiselect("Chọn từ khoá để lọc bài viết:", category_articles['tags'].unique())

# if selected_keywords:
#     category_articles = category_articles[category_articles['tags'].isin(selected_keywords)]

# # Show articles in selected category and allow article selection
# if not category_articles.empty:
#     # Allow multi-method recommendation
#     recommendation_methods = st.multiselect(
#         "Chọn cách đề xuất bài viết:",
#         ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
#     )

#     selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)

#     if selected_article_title:
#         selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

#         st.header(f"{selected_article['title']}")
#         st.write(f"**Tác giả**: {selected_article['author_name']}")
#         st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
#         st.write(f"**Từ khoá**: {selected_article['tags']}")
#         st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
#         st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")

#         # Add to history
#         st.session_state.history.append(selected_article['title'])

#         # Step 5: Recommend based on the selected methods
#         def recommend_similar_articles(selected_article, category_articles, methods):
#             vectorizer = TfidfVectorizer(stop_words='english')
#             recommended_articles_all = pd.DataFrame()

#             for method in methods:
#                 if method == "Dựa trên bài viết chi tiết":
#                     descriptions = category_articles['description'].tolist()
#                     descriptions.append(selected_article['description'])
#                     tfidf_matrix = vectorizer.fit_transform(descriptions)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                 elif method == "Dựa trên từ khóa":
#                     keywords = category_articles['tags'].tolist()
#                     keywords.append(selected_article['tags'])
#                     tfidf_matrix = vectorizer.fit_transform(keywords)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                 elif method == "Dựa trên tác giả":
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles[category_articles['author_name'] == selected_article['author_name']]])

#                 elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
#                     authors_description = category_articles['author_description'].tolist()
#                     authors_description.append(selected_article['author_description'])
#                     tfidf_matrix = vectorizer.fit_transform(authors_description)
#                     cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                     cosine_sim = cosine_sim.flatten()
#                     similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                     recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#             recommended_articles_all = recommended_articles_all.drop_duplicates(subset="title")
#             return recommended_articles_all

#         recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_methods)

#         # Display recommended articles
#         if not recommended_articles.empty:
#             st.subheader("Các bài viết tương tự:")
#             for _, related_article in recommended_articles.iterrows():
#                 if related_article['title'] != selected_article['title']:
#                     st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")

#         # Button to continue after viewing recommendations
#         if st.button("Tiếp tục lựa chọn bài viết khác"):
#             st.rerun()  # This will refresh the app and allow the user to continue choosing

# else:
#     st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")

# # Display history of selected articles
# if st.session_state.history:
#     st.subheader("Lịch sử chọn bài viết:")
#     for entry in st.session_state.history:
#         st.write(f"- {entry}")

# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# # Step 1: Choose a category
# category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# # Step 2: Display all articles in the selected category
# def get_articles_by_category(category):
#     return df[df['category'] == category]

# category_articles = get_articles_by_category(category_input)

# # Step 3: User selects a recommendation method
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Step 4: Add multiselect for filtering articles by keywords
# selected_keywords = st.multiselect("Chọn từ khoá để lọc bài viết:", category_articles['tags'].unique())

# if selected_keywords:
#     category_articles = category_articles[category_articles['tags'].isin(selected_keywords)]

# # Show articles in selected category and allow article selection
# if not category_articles.empty:
#     # Step 5: First show article selection
#     selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)

#     if selected_article_title:
#         selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

#         # Display the selected article details
#         st.header(f"{selected_article['title']}")
#         st.write(f"**Tác giả**: {selected_article['author_name']}")
#         st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
#         st.write(f"**Từ khoá**: {selected_article['tags']}")
#         st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
#         st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")

#         # Add the selected article to history
#         if selected_article['title'] not in st.session_state.history:
#             st.session_state.history.append(selected_article['title'])

#         # Step 6: Show recommendation methods after article selection
#         recommendation_methods = st.multiselect(
#             "Chọn các phương thức khuyến nghị:",
#             ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
#         )

#         # Button to trigger recommendations
#         if st.button("Khuyến nghị"):
#             # Step 7: Recommend based on the selected methods
#             def recommend_similar_articles(selected_article, category_articles, methods):
#                 vectorizer = TfidfVectorizer(stop_words='english')
#                 recommended_articles_all = pd.DataFrame()

#                 for method in methods:
#                     if method == "Dựa trên bài viết chi tiết":
#                         descriptions = category_articles['description'].tolist()
#                         descriptions.append(selected_article['description'])
#                         tfidf_matrix = vectorizer.fit_transform(descriptions)
#                         cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                         cosine_sim = cosine_sim.flatten()
#                         similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                         recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                     elif method == "Dựa trên từ khóa":
#                         keywords = category_articles['tags'].tolist()
#                         keywords.append(selected_article['tags'])
#                         tfidf_matrix = vectorizer.fit_transform(keywords)
#                         cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                         cosine_sim = cosine_sim.flatten()
#                         similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                         recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                     elif method == "Dựa trên tác giả":
#                         recommended_articles_all = pd.concat([recommended_articles_all, category_articles[category_articles['author_name'] == selected_article['author_name']]])

#                     elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
#                         authors_description = category_articles['author_description'].tolist()
#                         authors_description.append(selected_article['author_description'])
#                         tfidf_matrix = vectorizer.fit_transform(authors_description)
#                         cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#                         cosine_sim = cosine_sim.flatten()
#                         similar_articles_idx = cosine_sim.argsort()[-5:][::-1]
#                         recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

#                 recommended_articles_all = recommended_articles_all.drop_duplicates(subset="title")
#                 return recommended_articles_all

#             recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_methods)

#             # Display recommended articles
#             if not recommended_articles.empty:
#                 st.subheader("Các bài viết tương tự:")
#                 for _, related_article in recommended_articles.iterrows():
#                     if related_article['title'] != selected_article['title']:
#                         st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")

#             # Button to continue after viewing recommendations
#             if st.button("Tiếp tục lựa chọn bài viết khác"):
#                 st.experimental_rerun()  # This will refresh the app and allow the user to continue choosing

# else:
#     st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")

# # Display history of selected articles (only once)
# if st.session_state.history:
#     st.subheader("Lịch sử chọn bài viết:")
#     for entry in st.session_state.history:
#         st.write(f"- {entry}")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv(r"D:\ds300-code\DS300\ViNewRec\article_data_.csv")

# Convert 'publish_date' column to datetime and remove timezone info (if exists)
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce').dt.tz_localize(None)

# Step 1: Choose a category
category_input = st.selectbox("Chọn thể loại bài viết:", df['category'].unique())

# Step 2: Display all articles in the selected category
def get_articles_by_category(category):
    return df[df['category'] == category]

category_articles = get_articles_by_category(category_input)

# Step 3: User selects a recommendation method
if 'history' not in st.session_state:
    st.session_state.history = []

# Step 4: Add multiselect for filtering articles by keywords
selected_keywords = st.multiselect("Chọn từ khoá để lọc bài viết:", category_articles['tags'].unique())

if selected_keywords:
    category_articles = category_articles[category_articles['tags'].isin(selected_keywords)]

# Step 5: Add date filter - Date range selection
date_range = st.date_input("Chọn ngày bắt đầu và ngày kết thúc", 
                          value=(category_articles['publish_date'].min(), category_articles['publish_date'].max()))

start_date, end_date = date_range

# Ensure start and end dates are also naive (without timezone)
start_date = pd.to_datetime(start_date).normalize()  # normalize to remove time part
end_date = pd.to_datetime(end_date).normalize()  # normalize to remove time part

# Filter articles by date range
category_articles = category_articles[(category_articles['publish_date'] >= start_date) & 
                                      (category_articles['publish_date'] <= end_date)]

# Show filtered articles
if not category_articles.empty:
    # Allow multi-method recommendation
    recommendation_methods = st.multiselect(
        "Chọn cách đề xuất bài viết:",
        ("Dựa trên bài viết chi tiết", "Dựa trên từ khóa", "Dựa trên tác giả", "Dựa trên mô tả lĩnh vực hoạt động của tác giả")
    )

    # Step 6: Show article selection first, and then the "Khuyến nghị" button
    selected_article_title = st.selectbox("Chọn bài viết để xem chi tiết:", category_articles['title'].values)
    selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

    st.header(f"{selected_article['title']}")
    st.write(f"**Ngày**: {selected_article['publish_date']}")
    st.write(f"**Tác giả**: {selected_article['author_name']}")
    st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
    st.write(f"**Từ khoá**: {selected_article['tags']}")
    st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
    st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")
    # Add functionality to recommend articles only after selecting a method and clicking the button
    if st.button("Khuyến nghị") and selected_article_title:
        # selected_article = category_articles[category_articles['title'] == selected_article_title].iloc[0]

        # st.header(f"{selected_article['title']}")
        # st.write(f"**Ngày**: {selected_article['publish_date']}")
        # st.write(f"**Tác giả**: {selected_article['author_name']}")
        # st.write(f"**Lĩnh vực**: {selected_article['author_description']}")
        # st.write(f"**Từ khoá**: {selected_article['tags']}")
        # st.write(f"**Dẫn nhập nội dung**: {selected_article['description']}")
        # st.write(f"[Đọc bài đầy đủ]({selected_article['url']})")

        # Add to history
        st.session_state.history.append(selected_article['title'])

        # Step 7: Recommend based on the selected methods
        def recommend_similar_articles(selected_article, category_articles, methods):
            vectorizer = TfidfVectorizer(stop_words='english')
            recommended_articles_all = pd.DataFrame()

            for method in methods:
                if method == "Dựa trên bài viết chi tiết":
                    descriptions = category_articles['description'].tolist()
                    descriptions.append(selected_article['description'])
                    tfidf_matrix = vectorizer.fit_transform(descriptions)
                    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                    cosine_sim = cosine_sim.flatten()
                    similar_articles_idx = cosine_sim.argsort()[-10:][::-1]
                    recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

                elif method == "Dựa trên từ khóa":
                    keywords = category_articles['tags'].tolist()
                    keywords.append(selected_article['tags'])
                    tfidf_matrix = vectorizer.fit_transform(keywords)
                    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                    cosine_sim = cosine_sim.flatten()
                    similar_articles_idx = cosine_sim.argsort()[-10:][::-1]
                    recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

                elif method == "Dựa trên tác giả":
                    recommended_articles_all = pd.concat([recommended_articles_all, category_articles[category_articles['author_name'] == selected_article['author_name']]])

                elif method == "Dựa trên mô tả lĩnh vực hoạt động của tác giả":
                    authors_description = category_articles['author_description'].tolist()
                    authors_description.append(selected_article['author_description'])
                    tfidf_matrix = vectorizer.fit_transform(authors_description)
                    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                    cosine_sim = cosine_sim.flatten()
                    similar_articles_idx = cosine_sim.argsort()[-10:][::-1]
                    recommended_articles_all = pd.concat([recommended_articles_all, category_articles.iloc[similar_articles_idx]])

            recommended_articles_all = recommended_articles_all.drop_duplicates(subset="title")
            return recommended_articles_all

        recommended_articles = recommend_similar_articles(selected_article, category_articles, recommendation_methods)

        # Display recommended articles
        if not recommended_articles.empty:
            st.subheader("Các bài viết tương tự:")
            for _, related_article in recommended_articles.iterrows():
                if related_article['title'] != selected_article['title']:
                    st.write(f"- {related_article['title']} ([Đọc thêm]({related_article['url']}))")

        # Button to continue after viewing recommendations
        if st.button("Tiếp tục lựa chọn bài viết khác"):
            st.experimental_rerun()  # This will refresh the app and allow the user to continue choosing

else:
    st.write(f"Không có bài viết nào trong thể loại '{category_input}'.")

# Display history of selected articles (only once)
if st.session_state.history:
    st.subheader("Lịch sử chọn bài viết:")
    for entry in st.session_state.history:
        st.write(f"- {entry}")
