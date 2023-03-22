import praw
import time
import json
import os

reddit = praw.Reddit(client_id='iGBXSi7IenvrAFA2TDYVSg',
                     client_secret='_ine6fSfIVjBFaGDiFpXWaKVmFEqNQ',
                     password='Mus@dac20.',
                     user_agent='https',
                     username='musadac')



# Set the subreddit you want to scrape data from
subreddit = reddit.subreddit('python')

# Set the number of posts you want to scrape
num_posts = 1

# Set the name of the output file
output_file = 'reddit_data.json'

# Create an empty list to store the scraped data
data = []

# Scrape data from the subreddit
for post in subreddit.new(limit=num_posts):
    post_dict = {}
    post_dict['title'] = post.title
    post_dict['score'] = post.score
    post_dict['id'] = post.id
    post_dict['url'] = post.url
    post_dict['comms_num'] = post.num_comments
    post_dict['created'] = post.created
    post_dict['body'] = post.selftext
    data.append(post_dict)

    # Wait for 2 seconds to avoid overloading the Reddit API
    time.sleep(2)

# Save the data to a JSON file
with open(output_file, 'w') as f:
    json.dump(data, f)

# Print a message to indicate the data has been saved
print(f'{len(data)} posts have been saved to {os.getcwd()}/{output_file}')