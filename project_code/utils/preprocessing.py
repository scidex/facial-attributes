# Utils for preprocessing the data

import pandas as pd

import matplotlib.image as mpimg

# Filters the DataFrame into a DataFrame with people containing images and
# a DataFrame with people that do not contain images.
# Returns a tuple containing a DataFrame with and without images respectively.
def filter_without_images(df: pd.DataFrame, image_dir: str):
    # Create a copy for the DF without images.
    without_images = pd.DataFrame().reindex_like(df)
    without_images = without_images.iloc[0:0]
    
    # Create a copy for the DF with images.
    with_images = df.copy()
    
    no_img_count = 0
    
    for index, row in df.iterrows():
        image_filename = row['id'] + '.jpg'
        img = mpimg.imread(image_dir + image_filename)
        
        # Give feedback on the progress.
        if index % 2500 == 0:
            print('Currently at row number: {},'.format(index) 
                  + ' number of cases without image: {}'.format(no_img_count))
        if img.shape[2] == 4:
            no_img_count += 1
            without_images = without_images.append(df.iloc[index])
            with_images = with_images.drop(df.index[index])

    return with_images, without_images
