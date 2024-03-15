import pandas as pd

def load_dataset(dataset_name, dr_method=None, n_components=8):
    palette = None

    if dataset_name == "Salinas":
        if dr_method is not None:
            df = pd.read_csv(f'Datasets/{dr_method}/{dataset_name}/Salinas_{dr_method}_{n_components}.csv')
            img = df.iloc[:, :-1]
            gt = df.iloc[:, -1:]
                
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        label_values = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]

        ignored_labels = [0]
                            
    # Filter NaN out
    # nan_mask = np.isnan(img.sum(axis=-1))
    # if np.count_nonzero(nan_mask) > 0:
    #     print(
    #         "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
    #     )
    # img[nan_mask] = 0
    # gt[nan_mask] = 0
    # ignored_labels.append(0)

    # ignored_labels = list(set(ignored_labels))
    # # Normalization
    # img = np.asarray(img, dtype="float32")
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette