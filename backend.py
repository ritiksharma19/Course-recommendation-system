import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

models = ("Course Similarity",
          "User Profile",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

# Random State
rs = 23

def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def load_course_genre():
    return pd.read_csv("course_genre.csv")

def load_user_profile():
    return pd.read_csv("user_profile.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict



def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def get_recommendation_scores(user_id, course_genres_df, user_profile_df, ratings_df):
            # Get the user profile data for the current user
            test_user_profile = user_profile_df[user_profile_df['user'] == user_id]
            #Get the user vector for the current user
            test_user_vector = test_user_profile.iloc[0, 1:].values


            #Get the known course ids for the current user
            enrolled_courses = ratings_df[ratings_df['user'] == user_id]['item'].to_list()

            #Calculate the unknown course ids
            all_courses = set(course_genres_df['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_courses)

            #Filter the course_genres_df to include only unknown courses
            unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
            unknown_course_ids = unknown_course_df['COURSE_ID'].values

            # Calculate recommendation scores using dot product
            recommendation_score = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

            return recommendation_score, unknown_course_ids


def combine_cluster_label(user_ids, labels):
  labels_df = pd.DataFrame(labels)
  cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
  cluster_df.columns = ['user', 'cluster']
  return cluster_df


def principal_component_analysis_user_profile(df):
    features = df.loc[:, df.columns != 'user']
    user_id = df.loc[:, df.columns == 'user']
    feature_names = list(df.columns.values)[1:]

    # Finding optimized n_components for PCA
    pca = PCA(n_components=None)
    pca.fit(features)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    desired_variance = 0.95
    n_components = np.argmax(cumulative_variance_ratio > desired_variance)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)
    pca_df = pd.DataFrame(data=components)
    pca_df.index = df['user']
    return pca_df


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    
    if "selected_user_id" in params:
        selected_user_id = params["selected_user_id"]
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            users.clear()
            courses.clear()
            scores.clear()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        if model_name == models[1]:
            # Load resources
            users.clear()
            courses.clear()
            scores.clear()
            course_genres_df = load_course_genre()
            user_profile_df = load_user_profile()
            ratings_df = load_ratings()

            # Calculate recommendation scores using dot product
            recommendation_score, unknown_course_ids = get_recommendation_scores(2, course_genres_df, user_profile_df, ratings_df)

            for i in range(0, len(unknown_course_ids)):
                score = recommendation_score[i]

                if score >= 10.0:
                    users.append(selected_user_id)
                    courses.append(unknown_course_ids[i])
                    scores.append(recommendation_score[i])
        if model_name == models[2]:
            users.clear()
            courses.clear()
            scores.clear()
            user_profile_df = load_user_profile()
            ratings_df = load_ratings()
            features = user_profile_df.loc[:, user_profile_df.columns != 'user']
            user_id = user_profile_df.loc[:, user_profile_df.columns == 'user']

            # Perform K-Means clustering
            inertia_values = []
            for n_cluster in range(1,params['cluster_no']):
                kmeans = KMeans(n_clusters=n_cluster, random_state = rs)
                kmeans.fit(features)
                inertia_values.append(kmeans.inertia_)
            
            kl = KneeLocator(range(1,params['cluster_no']), inertia_values, curve='convex', direction='decreasing')
            n_cluster = kl.elbow

            cluster_lables = [None] * len(user_id)

            kmeans = KMeans(n_clusters=n_cluster, random_state=rs)
            kmeans.fit(principal_component_analysis_user_profile(user_profile_df))

            cluster_lables = kmeans.labels_

            cluster_df = combine_cluster_label(user_id, cluster_lables)
            
            test_users_df = ratings_df[['user', 'item']]

            test_user_labelled = pd.merge(test_users_df, cluster_df, on='user')   
      

            # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
            courses_cluster = test_user_labelled[['item', 'cluster']]

            # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
            courses_cluster['count'] = [1] * len(courses_cluster)

            
            # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
            # and resetting the index to make the result more readable
            courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()


            # Popular courses
            popular_courses = courses_cluster_grouped[courses_cluster_grouped['enrollments'] > 100]

            user_subset = test_user_labelled[test_user_labelled['user'] == int(selected_user_id)]

            enrolled_courses = user_subset['item'].tolist()

            recommendations = popular_courses[~popular_courses['item'].isin(enrolled_courses)]['item'].tolist()

            for i in recommendations[:10]:
                users.append(user_id)
                courses.append(i)
                scores.append(0)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
