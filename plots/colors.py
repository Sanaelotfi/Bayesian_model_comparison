import seaborn as sns


def get_color(lst):
    return list(map(lambda x: x / 255., lst))

color1 = get_color([128, 179, 255])
color2 = get_color([26, 26, 26])#"k"
color3 = get_color([147, 157, 172])

color4 = get_color([255, 204, 249])
# temp_color = get_color([255, 238, 170])
color5 = get_color([222, 135, 135])
# pca_color = get_color([255, 238, 170])
# pca_color = get_color([175, 233, 198])
color6 = get_color([135, 222, 170])
color7 = get_color([221, 233, 175])
color8 = get_color([238, 170, 255])
color9 = get_color([233, 221, 175])

color10 = get_color([255, 42, 42])
color11 = get_color([55, 200, 113])
color12 = get_color([55, 113, 200])

sanae_colors = [sns.cubehelix_palette(as_cmap=False)[i] for i in range(6)]
