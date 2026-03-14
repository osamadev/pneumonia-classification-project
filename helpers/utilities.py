def show_samples(label):

    folder = os.path.join(train_dir, label)
    files = os.listdir(folder)[:5]

    plt.figure(figsize=(12,3))

    for i, file in enumerate(files):

        img = Image.open(os.path.join(folder, file))

        plt.subplot(1,5,i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.suptitle(label)
    plt.show()


def compute_brightness(folder, sample_size=300):

    brightness = []

    files = os.listdir(folder)[:sample_size]

    for file in files:

        img = Image.open(os.path.join(folder, file)).convert("L")
        img_array = np.array(img)

        brightness.append(img_array.mean())

    return brightness
