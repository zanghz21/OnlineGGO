
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


def gen_video(base_dir, key_word):
    file_list = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    image_files_num = len([f for f in file_list if key_word in f])
    image_files = [os.path.join(base_dir, f"{i}_{key_word}.png") for i in range(1, image_files_num+1)]

    # Set the frame size (width, height) based on your images
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    save_dir = os.path.join(base_dir, "video")
    os.makedirs(save_dir, exist_ok=True)
    output_video = cv2.VideoWriter(os.path.join(save_dir, f"{key_word}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for image in image_files:
        frame = cv2.imread(image)
        output_video.write(frame)

    # Release the VideoWriter object
    output_video.release()

    print("Video created successfully!")

        
        
def gen_full_video(base_dir, total_time=50):
    arr_name = [f"gg{i}" for i in range(5, 10)] + [f"traffic{i}" for i in range(5)]
    save_dir = os.path.join(base_dir, "full_vis")
    os.makedirs(save_dir, exist_ok=True)
    for t in tqdm(range(total_time)):
        
        images = [os.path.join(base_dir, f"{t+1}_{name}.png") for name in arr_name]

        assert len(images) == 10

        image_arrays = [np.array(Image.open(img)) for img in images]
        image_shape = image_arrays[0].shape
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(image_shape[1]//100 *5, image_shape[0]//100 *2))

        for i, ax in enumerate(axes.flat):
            ax.imshow(image_arrays[i], aspect='auto')
            ax.axis('off')
            
        plt.savefig(os.path.join(save_dir, f"{t+1}.png"))
        plt.close()
    
    full_image_files = [os.path.join(save_dir, f"{t+1}.png") for t in range(total_time)]
    frame = cv2.imread(full_image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    save_dir = os.path.join(base_dir, "video")
    os.makedirs(save_dir, exist_ok=True)
    output_video = cv2.VideoWriter(os.path.join(save_dir, f"full_vis.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for image in full_image_files:
        frame = cv2.imread(image)
        output_video.write(frame)

    # Release the VideoWriter object
    output_video.release()
    

def main():
    base_dir = ""
    key_words = ["task"] + [f"gg{i}" for i in range(5, 10)] + [f"traffic{i}" for i in range(5)]
    # for key_word in key_words:
    #     gen_video(base_dir, key_word)
    gen_full_video(base_dir)

if __name__ == "__main__":
    main()