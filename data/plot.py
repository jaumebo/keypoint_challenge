import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import argparse

KEYPOINT_NAMES = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
                  "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
                  "right_ankle", "left_hip", "left_knee", "left_ankle",
                  "right_eye", "left_eye", "right_ear", "left_ear"]
SKELETON = [[16, 14], [14, 0], [15, 0], [15, 17], [0, 1], [1, 2],
            [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13]]

def plot_video(path_to_csv, output_name='example.mp4',
               plot_labels=False, border_size=0.5):
    data = pd.read_csv(path_to_csv, header=None, names=['frame_id', 'keypoint_id', 'x', 'y', 'score'])
    fig, ax = plt.subplots(figsize=(10, 10))
    data['x'] = ((data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min()))
    data['y'] = ((data['y'] - data['y'].min())/ (data['y'].max() - data['y'].min()))
    ax.set_xlim(0 - border_size, 1 + border_size)
    ax.set_ylim(0 - border_size, 1 + border_size)
    def animate(index):
        plt.cla()
        ax.set_xlim(0 - border_size, 1 + border_size)
        ax.set_ylim(0 - border_size, 1 + border_size)
        ax.set_ylim(ax.get_ylim()[::-1])
        x = data[data['frame_id']==index]['x'].values
        y = data[data['frame_id']==index]['y'].values
        keypoint_ids = list(data[data['frame_id']==index]['keypoint_id'].values)
        ax.scatter(x, y)
        if plot_labels:
            for i, row in data[data['frame_id']==index].iterrows():
                ax.annotate(KEYPOINT_NAMES[int(row['keypoint_id'])-1],
                            (row['x'], row['y']), color='magenta')
        for pair_i, pair_j in SKELETON:
            pair_i += 1
            pair_j += 1
            if pair_i in keypoint_ids and pair_j in keypoint_ids:
                ax.plot([x[keypoint_ids.index(pair_i)],
                         x[keypoint_ids.index(pair_j)]],
                        [y[keypoint_ids.index(pair_i)],
                         y[keypoint_ids.index(pair_j)]], color='cyan',
                        linestyle='-', linewidth=5)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        return ax
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=70, repeat=True)
    ani.save(output_name, writer=writer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an mp4 file from the provided .csv')
    parser.add_argument('--input_path', type=str, required=True,
                        help='path to input .csv e.g 000_pick_up.csv')
    parser.add_argument('--output_path', default='example.mp4', type=str,
                        help='path to output video, should include .mp4 extension')
    parser.add_argument('--plot_labels', default=1, type=int,
                        help='1 or 0, plot the labels')
    parser.add_argument('--border_size', default=0.5, type=float,
                        help='increase the border, border size of 1 is doubling the border')
    args = parser.parse_args()
    plot_video(path_to_csv=args.input_path,
               output_name=args.output_path,
               plot_labels=args.plot_labels,
               border_size=args.border_size)
