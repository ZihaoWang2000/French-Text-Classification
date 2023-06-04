import streamlit as st
import joblib
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertModel
import torch
import torch.nn as nn
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from youtube_transcript_api import YouTubeTranscriptApi
import tqdm
import re
import os
import pickle

device = torch.device('mps')
model_best = joblib.load('best_model.pkl')

# Cambert model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base', num_labels=6).to(device)

def bert_feature(data, **kwargs):
    # Tokenize and encode input texts
    input_ids = [tokenizer.encode(text, add_special_tokens=True, **kwargs) for text in data]

    # Extract BERT features for each input ID
    features = []
    with torch.no_grad():
        for input_id in tqdm.tqdm(input_ids):
            # Convert input ID to tensor
            input_tensor = torch.tensor(input_id).unsqueeze(0).to(device)

            # Extract BERT features for this input ID
            input_embeds = model.embeddings(input_tensor)
            feature = model(inputs_embeds=input_embeds)[0][:, 0, :].cpu().numpy()

            # Add feature to list of all features
            features.append(feature)

    # Concatenate features from all inputs
    feature_data = np.concatenate(features, axis=0)

    torch.cuda.empty_cache()

    return feature_data


SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
def youtube_authenticate():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "credentials.json"
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)

def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()

def get_video_infos(video_response):
    items = video_response.get("items")[0]
    # get the snippet, statistics & content details from the video response
    snippet = items["snippet"]
    statistics = items["statistics"]
    content_details = items["contentDetails"]
    # get infos from the snippet
    channel_title = snippet["channelTitle"]
    title = snippet["title"]
    description = snippet["description"]
    thumbnails_url = snippet['thumbnails']['default']['url']
    return title, description, thumbnails_url

def search(youtube, **kwargs):
    return youtube.search().list(
        part="snippet",
        type="video",
        **kwargs
    ).execute()

def retrieve_video_list(keyword):
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    
    # search for the query 'python' and retrieve results items only
    response = search(youtube, q=keyword, maxResults=50, relevanceLanguage="FR", videoCaption="closedCaption")
    items = response.get("items")
    df_video = pd.DataFrame(columns=['video url', 'title', 'description', 'thumbnails url', 'caption', 'difficulty'])
    for item in tqdm.tqdm(items):
        # get the video ID
        video_id = item["id"]["videoId"]
        # get the video details
        video_response = get_video_details(youtube, id=video_id)
        title, description, thumbnails_url = get_video_infos(video_response)

        # assigning srt variable with the list
        # of dictionaries obtained by the get_transcript() function
        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
            caption = '. ' .join(item['text'] for item in srt)
            caption = caption.replace('\n', ' ')
            video = {'video url': 'https://www.youtube.com/watch?v=' + video_id, 'title': title, 'description': description, 'thumbnails url': thumbnails_url, 'caption':caption}
            df_video = df_video.append(video, ignore_index=True)
        
        except:
            continue
    
    return df_video

def predictor(keyword, level):
    # Retrieve the videos using keyword
    df_video = retrieve_video_list(keyword)

    # Extract text features through bert
    test_features = bert_feature(df_video['caption'], max_length=256)

    # Predict difficulty
    pred_difficulty = model_best.predict(test_features)
    df_video['difficulty'] = pd.Series(pred_difficulty).map({0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'C1', 5:'C2'})

    # Match the difficulty with input
    df_result = df_video[df_video['difficulty'] == level].reset_index()

    return df_result

# Create Input and Selector
keyword_input = st.text_input('Keyword:')
level_dropdown = st.selectbox('Level:', ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])

# Create button
show_videos_button = st.button('Display Recommended Youtube Videos')


# Define the function for button click event
def show_videos():
    st.empty()

    # Get the value of keyword and level
    keyword = keyword_input
    level = level_dropdown

    # Display the loading icon
    with st.spinner('Loading...'):
        # Execute the predictor function and get the result
        df_result = predictor(keyword, level)

    if len(df_result) != 0:
        if len(df_result) == 1:
            st.write('There is 1 video recommended for you!')
        else:
            st.write('There are {} videos recommended for you!'.format(len(df_result)))
            
        st.write('\n')

        for i in range(len(df_result)):
            st.write(str(i + 1) + '. Title: ' + df_result['title'][i])
            st.write('\nurl: ' + df_result['video url'][i])
            st.image(df_result['thumbnails url'][i])
            st.write('\n')
            st.write('\n')
    else:
        st.write('There is no match for your input! Please change your keyword or level! (Note: Most videos are above C1 level)')

# Connect the button click event and the function
if show_videos_button:
    show_videos()
