import albumentations as A



weak_transforms = A.Compose([ 
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Transpose(),
    A.RandomBrightnessContrast()
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]), 
) 
strong_transforms = A.Compose(
    [
        A.Posterize(), 
        A.Equalize(), 
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(), 
        A.RandomShadow(), 
    ]
)


