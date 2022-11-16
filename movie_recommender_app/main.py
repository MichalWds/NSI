#Authors: Michal Wadas, Karol Kuchnio
import argparse
import json
import numpy as np


def cli_arg_parser():
    parser = argparse.ArgumentParser(description='Engine to compute similarity score')
    choices = ['euclidean', 'pearson', 'Euclidean', 'Pearson', 'EUCLIDEAN', 'PEARSON']
    parser.add_argument('--user','-u', dest='user', required=True,
                        help='User')
    parser.add_argument('--similarity-type','-sc', dest="similarity_type", required=True,
                        choices=choices, type=make_type(choices),help='Please provide one of the Similarity metric.')
    return parser

#Function to extract data from json & validate we have proper data for input user
def extract_data(args, ratings_data):
    with open(ratings_data, 'r') as f:
        data = json.loads(f.read())
    if args.user not in data:
        raise TypeError('User % s not found in the data collection' % args.user)
    return data

#Function to handle case sensitive similarity type
def make_type(choices):
    def find_choice(choice):
        for key, item in enumerate(choice.lower() for choice in choices):
            if choice.lower() == item:
                return choices[key]
        else:
            return choice

    return find_choice

#Function used to find common movies
def get_common_movies(dataset, user_1, user_2):
    return {item: 1 for item in dataset[user_1] if item in dataset[user_2]}


#Function to calculate the euclidean distance score between two users(user1 & user2)
def euclidean_score(dataset, user_1, user_2):
    compute_squared_diff = [np.square(dataset[user_1][item] - dataset[user_2][item]) for item in dataset[user_1] if item in dataset[user_2]]
    return 1 / (1 + np.sqrt(np.sum(compute_squared_diff)))


#Function to calculate the pearson correlation score between two users(user1 & user2)
def count_pearson_score(dataset, user_1, user_2):
    #Movies rated by both user1 and user2
    common_movies = get_common_movies(dataset, user_1, user_2)
    num_ratings = len(common_movies)

    #Sum of ratings for all the common movies
    user1_sum = np.sum([dataset[user_1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user_2][item] for item in common_movies])

    #Suares of ratings for all the common movies
    user1_sqrt_sum = np.sum([np.square(dataset[user_1][item]) for item in common_movies])
    user2_sqrt_sum = np.sum([np.square(dataset[user_2][item]) for item in common_movies])

    #Sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user_1][item] * dataset[user_2][item] for item in common_movies])

    #Calculate Pearson correlation score
    u_xy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    u_xx = user1_sqrt_sum - np.square(user1_sum) / num_ratings
    u_yy = user2_sqrt_sum - np.square(user2_sum) / num_ratings

    return 0 if u_xx * u_yy == 0 else u_xy / np.sqrt(u_xx * u_yy)

#Function to get duplicates in list and averages of the ratings
def get_avg_duplicates(movies):
    temp = []
    for movie in movies:
        if movie in temp:
            continue
        movie_list = [i for i in movies if movie[0] == i[0]]
        avg_movie = [movie_list[0][0], 0]
        for i in movie_list:
            avg_movie[1] += i[1]
        avg_movie[1] = avg_movie[1] / len(movie_list)
        if avg_movie not in temp:
            temp.append(avg_movie)
    return temp

if __name__ == '__main__':
    args = cli_arg_parser().parse_args()
    print("Similarity % s for user  % s was chosen" % (args.similarity_type, args.user))
    ratings_data = 'ratings.json'
    data = extract_data(args, ratings_data)
    temp = []
    for other_user in data:
        ratio = 0
        if args.similarity_type == 'euclidean':
            ratio = euclidean_score(data, args.user, other_user)
        if args.similarity_type == 'pearson':
            ratio = count_pearson_score(data, args.user, other_user)
        distance = len(get_common_movies(data, args.user, other_user))

        #skip if there is no similar movies between users
        if other_user == args.user or distance == 0:
            continue
        temp.extend([movie, ratio * distance * data[other_user][movie]] 
            for movie in data[other_user] if movie not in data[args.user].keys())

    temp = get_avg_duplicates(temp)
    temp.sort(key=lambda x: x[1])

    print("Recomended movie for for you:")
    for i in temp[-5:]:
        print(i[0])
    print("\n")
    print("I would rather not recommend theose ones:")
    for i in temp[:5]:
        print(i[0])