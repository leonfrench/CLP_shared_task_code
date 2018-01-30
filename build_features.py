import os
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from empath import Empath
import config
# from sklearn import preprocessing


def process_body(text):
    """
    Helper function to process/clean the post body.
    """
    # if text != None:
    if text is not None:
        soup = BeautifulSoup(str(text), 'html.parser')
        try:
            soup.find('blockquote').decompose()
            contained_quote = True

        except AttributeError:
            contained_quote = False

        cleaned = soup.get_text()
        cleaned = unicodedata.normalize("NFKD", cleaned)

    return cleaned, contained_quote


def process_images(text):
    """
    Helper used to extract images from post body for a image feature column.
    """
    # if text != None:
    if text is not None:
        soup = BeautifulSoup(str(text), 'html.parser')
        img = soup.img
        try:
            image = img['title']
            return image
        except (TypeError, KeyError):
            # print(img)
            pass


def convert_dates(datetime_col):
    # doesnt work yet
    # this function should be able to take the datetime column and return 3 new
    # columns for month, season and time of day all at once
    columns = []

    datetime_col = pd.to_datetime(datetime_col)
    transforms = [
            lambda x: x.month,
            lambda x: x.hour]

    for transform in transforms:
        columns.append(datetime_col.apply(transform))

    return pd.DataFrame(columns).T


def convert_month(datetime_col):
    datetime_col = pd.to_datetime(datetime_col)
    month_col = datetime_col.apply(lambda x: x.month)
    return month_col


def convert_season(month_col):
    mapper = {
              1: 0,
              2: 0,
              3: 1,
              4: 1,
              5: 1,
              6: 2,
              7: 2,
              8: 2,
              9: 3,
              10: 3,
              11: 3,
              12: 0
              }
    return month_col.map(mapper)


def convert_hour(datetime_col):
    datetime_col = pd.to_datetime(datetime_col)
    month_col = datetime_col.apply(lambda x: x.hour)
    return month_col


# a few basic character level matches, inspired by:
# http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost
def count_character(text, target_character):
    if text is None:
        return 0
    return sum(1 for c in text if c == target_character)


def count_upper_characters(text):
    if text is None:
        return 0
    return sum(1 for c in text if c.isupper())


def process_thread(current_row, full_df, filter_for_staff=False, previous=False):
    """
    Helper function to find next post in thread returns previous post and next
    post, could be sped up to return all four at once
    """
    thread_posts = full_df[(full_df.thread == current_row['thread'])]
    if filter_for_staff:
        thread_posts = thread_posts[thread_posts.is_staff == True]

    if previous:
        filtered_posts = thread_posts[thread_posts.post_time < current_row['post_time']]
        filtered_posts = filtered_posts.sort_values(by='post_time',
                                                    ascending=False)
    else:  # find the following post
        filtered_posts = thread_posts[thread_posts.post_time > current_row['post_time']]
        filtered_posts = filtered_posts.sort_values(by='post_time',
                                                    ascending=True)

    if (filtered_posts.empty):
        return None

    closest_post = filtered_posts.iloc[0]
    return closest_post['post_id']


def convert_labels(labels):
    # sklearn mapper doesnt work on NaN?
    # encoder = preprocessing.LabelEncoder()
    # encoder.fit(labels.fillna(99))

    mapper = {
              'green': 0,
              'amber': 1,
              'red': 2,
              'crisis': 3
              }
    return labels.map(mapper)


def vader_polarity(text):
    """
    Convenience function to transform text body into 4 separate columns of
    vader sentiment score
    """
    score = vader.polarity_scores(text)
    return score['compound'], score['neg'], score['neu'], score['pos']

    
def create_empath_cats(text):
    try:
        cat_scores = lexicon.analyze(text, normalize=True)
    except Exception as e:
        print(e)
        return 0
    return pd.Series(cat_scores)


def create_empath_df(text_column):
    empath_df = text_column.apply(create_empath_cats)
    empath_df = empath_df.add_prefix('empath_')
    empath_df = empath_df.reset_index().rename(columns={'index': 'post_id'})
    return empath_df


def top_images(images, n_filter):
    """
    Filters out the unique image names to keep those that appear at least
    n times in the corpus.
    Returns a 1-hot encoded dataframe for presence of the top images.
    """
    top_images = images.value_counts()[images.value_counts() > n_filter].index
    filtered = images.where(images.isin(top_images))
    assert(filtered.shape == images.shape)

    encoded = pd.get_dummies(filtered, prefix='images')
    return encoded


def main():
    features_location = os.path.join(config.DATA_DIR,
                                     'interim',
                                     'processed_features.csv')

    if os.path.exists(features_location):
        print("-- processed_features.csv found locally")
        features_df = pd.read_csv(features_location)

    else:
        features_df = pd.read_csv(os.path.join(config.DATA_DIR,
                                               'interim',
                                               'all_posts_data.csv'), low_memory=True)
        # features_df = features_df.iloc[:1000, :]
        # append subject line to each text body to account for empty posts
        # df.body = df.subject.astype(str) + ' ' + df.body


        features_df['cleaned_body'], features_df['contained_quote'] = zip(*features_df['body'].apply(process_body))
        features_df['images'] = features_df['body'].apply(process_images)
        features_df['linear_label'] = convert_labels(features_df.label)  # keep both the label and the mapped label
        features_df['month'] = convert_month(features_df.post_time)
        features_df['season'] = convert_season(features_df.month)
        features_df['hour'] = convert_hour(features_df.post_time)
        features_df['is_first_post'] = features_df.parent_post.isnull()

        features_df['exclaim_count'] = features_df.cleaned_body.apply(lambda x: count_character(x, "!"))
        features_df['question_count'] = features_df.cleaned_body.apply(lambda x: count_character(x, "?"))
        features_df['at_mention_count'] = features_df.cleaned_body.apply(lambda x: count_character(x, "@"))
        features_df['capital_letter_count'] = features_df.cleaned_body.apply(lambda x: count_upper_characters(x))
        features_df['total_body_chars'] = features_df.cleaned_body.apply(lambda x: len(x))

        features_df['contained_img'] = features_df.images.notnull()
        # n_filter=5 leaves us with 30 of the top occuring images/emoticons
        one_hot_images = top_images(features_df['images'], n_filter=5)
        assert(one_hot_images.shape[0] == features_df.shape[0])
        features_df = pd.concat([features_df, one_hot_images], axis=1)

        # 1 hot encoding for board and author_rank - despite the name there is
        # no clear ranking of the author ranks that I see
        board_1hot = pd.get_dummies(features_df['board'], prefix='board_1hot')
        author_rank_1hot = pd.get_dummies(features_df['author_rank'],
                                          prefix='author_rank_1hot')
        assert(board_1hot.shape[0] == features_df.shape[0])
        assert(author_rank_1hot.shape[0] == features_df.shape[0])
        features_df = pd.concat([features_df, author_rank_1hot, board_1hot],
                                axis=1)

        features_df['vader_compound'], features_df['vader_neg'], features_df['vader_neu'], features_df['vader_pos'] = zip(*features_df.cleaned_body.apply(vader_polarity))

        labeled_posts = features_df[features_df.label.notnull() | features_df.predict_me]
        empath_df = create_empath_df(labeled_posts.set_index('post_id').cleaned_body)
        # to get empath features for all posts:
        #empath_df = create_empath_df(features_df.set_index('post_id').cleaned_body)
        
        empath_location = os.path.join(config.DATA_DIR,
                                       'interim',
                                       'empath_features.csv')

        print('-- Writing empath features to {} -- '.format(empath_location))
        empath_df.to_csv(empath_location, index=False)

        print('features_df shape before empath cats: {}'.format(features_df.shape))
        # merge on post_id
        features_df = features_df.merge(empath_df, how='left', on='post_id')
        print('features_df shape after adding empath cats: {}'.format(features_df.shape))

        features_df['following_post_id'] = features_df.apply(process_thread,
                                                             axis=1,
                                                             full_df=features_df)
        print("Done process thread 1")
        features_df['previous_post_id'] = features_df.apply(process_thread,
                                                            axis=1,
                                                            full_df=features_df,
                                                            previous=True)
        print("Done process thread 2")
        features_df['following_staff_post_id'] = features_df.apply(process_thread,
                                                                   axis=1,
                                                                   full_df=features_df,
                                                                   filter_for_staff=True)
        print("Done process thread 3")
        features_df['previous_staff_post_id'] = features_df.apply(process_thread,
                                                                  axis=1,
                                                                  full_df=features_df,
                                                                  previous=True,
                                                                  filter_for_staff=True)
        print("Done process thread 4")

        # drop unneeded columns for model
        # 'root_post' and 'thread' are the same
        drop_features = ['body', 'root_post']
        features_df = features_df.drop(drop_features, axis=1)

        print('-- Writing data to {} -- '.format(features_location))
        features_df.to_csv(features_location, index=False)

    return features_df


if __name__ == "__main__":
    vader = SentimentIntensityAnalyzer()
    lexicon = Empath()
    main()
