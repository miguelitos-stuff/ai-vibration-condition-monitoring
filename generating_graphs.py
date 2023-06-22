# All the plots should be square and
# remove the lines between data,
# I need the orange line,
# blue dots if it is classified correctly,
# and red dots if it is not predicted correctly.
# You can adjust the size of the dots (big enough)

import matplotlib.pyplot as plt
import numpy as np


for SENSOR in [False, True]:
    for MODEL_AMED in [True, False]:
        if MODEL_AMED:
            name = "AMED"
            title1 = "AMED Augmented Data "
        else:
            name = "FRED_30"
            title1 = "FRED Augmented Data "

        if SENSOR:
            sen_list = range(3, 10+1)
            n_type = "_test_sensor"
        else:
            sen_list = [-1, 0]
            n_type = ""

        title_font = {'family' : 'serif',
                #'weight' : 'bold',
                'size'   : 28}
        font = {'family' : 'serif',
                'size'   : 16}
        axis_font = {'family' : 'serif',
                'size'   : 10}

        for sen in sen_list:
            if sen == 0:
                sec_name = "_std"
                title = title1
            elif sen == -1:
                sec_name = "_std_noise_005"
                title = title1 + "with Noise"
            else:
                sec_name = f"_{sen}"
                title = title1 + f"Sensor {sen}"

            f_ = np.load(f'npy\\{name}{n_type}_f_.npy')
            t_list = np.arange(len(f_))
            t_point = t_list[f_ < 0.5][0]
            points = np.load(f'npy\\{name}{n_type}{sec_name}_.npy')
            blue_points = np.array([])
            blue_t = np.array([])
            red_points = np.array([])
            red_t = np.array([])

            for i in range(len(points)):
                blue = False
                t = t_list[i]
                point = points[i]
                if t < t_point:
                    if point > 0.5:
                        blue = True
                else:
                    if point < 0.5:
                        blue = True
                if blue:
                    blue_t = np.append(blue_t, np.array([t]))
                    blue_points = np.append(blue_points, np.array([point]))
                else:
                    red_t = np.append(red_t, np.array([t]))
                    red_points = np.append(red_points, np.array([point]))

            plt.figure(figsize=(10, 10))
            plt.xlim((0, len(t_list)))
            plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"], **axis_font)
            plt.xticks(**axis_font)
            plt.grid(color='0.8', linestyle='-', linewidth=0.5)
            plt.plot(blue_t, blue_points, ".", color="green", markersize=10)
            plt.plot(red_t, red_points, ".", color="red", markersize=10)
            plt.plot(t_list, f_, color="blue", linewidth=3)
            plt.title(title, **title_font)
            plt.ylabel("Probability [-]", **font)
            plt.xlabel("Images [-]", **font)

            plt.savefig("graphs3\\" + f"{name}{n_type}{sec_name}_" + ".png")

            print(title, len(red_points)/len(points)*100)





