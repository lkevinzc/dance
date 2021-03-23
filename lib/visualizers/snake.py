from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
import mpl_toolkits.axes_grid1 as axes_grid1

mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(
            img_utils.unnormalize_img(batch['inp'][0], mean,
                                      std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([[31, 119, 180], [255, 127, 14], [46, 160, 44],
                           [214, 40, 39], [148, 103, 189], [140, 86, 75],
                           [227, 119, 194], [126, 126, 126], [188, 189, 32],
                           [26, 190, 207]]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.show()

    def visualize_training_box(self, output, batch):
        inp = img_utils.bgr_to_rgb(
            img_utils.unnormalize_img(batch['inp'][0], mean,
                                      std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy(
        ) * snake_config.down_ratio

        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        # fig = plt.figure()
        # grid = axes_grid1.AxesGrid(
        #     fig,
        #     111,
        #     nrows_ncols=(1, 1),
        #     axes_pad=0.5,
        #     cbar_location="right",
        #     cbar_mode="each",
        #     cbar_size="10%",
        #     cbar_pad="2%",
        # )
        # grid[0].axis(False)
        # im1 = grid[0].imshow(inp, cmap='coolwarm_r', interpolation='nearest')

        colors = np.array([[31, 119, 180], [255, 127, 14], [46, 160, 44],
                           [214, 40, 39], [148, 103, 189], [140, 86, 75],
                           [227, 119, 194], [126, 126, 126], [188, 189, 32],
                           [26, 190, 207]]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            # print(poly.shape)
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=5)

            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min],
                    [y_min, y_max, y_max, y_min, y_min],
                    color='w',
                    linewidth=0.5)

        this_im_id = batch['meta']['img_id']

        # plt.savefig(
        #     'vis/dance/' + str(this_im_id.numpy()[0]) + '.png',
        #     bbox_inches='tight',
        #     pad_inches=0.0,
        #     dpi=200,
        # )

        plt.savefig('vis/snake/' + str(this_im_id.numpy()[0]) + '.png',
                    format='png')

    def visualize(self, output, batch):
        # self.visualize_ex(output, batch)
        self.visualize_training_box(output, batch)
