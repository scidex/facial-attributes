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
    
# Splits the given DataFrame into three subset DataFrames
# based on the provided fractions.
# Returns a tuple containing the training, validation, and test DataFrames
# respectively.
def split_data(data: pd.DataFrame, train_frac, val_frac):
    # Since we first split into train and test, calculate the relative
    # fraction of the validation set (such that it'll still containg val_frac
    # of the original dataset).
    val_frac = val_frac / train_frac

    # Split the data into train and test set.
    temp_train_set = data.sample(frac=train_frac, random_state=3072021)
    test_set = data.drop(temp_train_set.index)
    
    # Split the training data into train and validation set.
    train_set = temp_train_set.sample(frac=1 - val_frac, random_state=14072021)
    val_set = temp_train_set.drop(train_set.index)
    
    return train_set, val_set, test_set
