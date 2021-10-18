import datasets.ytvos_transforms as T

def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([300], max_size=540),
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_vistr_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([360], max_size=640),
            ])),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_vistr_photo_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([360], max_size=640),
            ])),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_vistr_short_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([360], max_size=640),
            ])),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=800),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([608], max_size=800),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_low_res_no_crop_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [280, 296, 312, 328, 344, 360]
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=640),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_low_res_photo_no_crop_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [280, 296, 312, 328, 344, 360]
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomResize(scales, max_size=640),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_low_res_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [280, 296, 312, 328, 344, 360]
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([360], max_size=640),
            ])),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_fixed_low_res_crop_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [280, 296, 312, 328, 344, 360]
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
            #     T.Compose([
            #         T.RandomResize([360], max_size=640),
            #         T.RandomResize([200, 250, 300]),
            #         T.RandomSizeCrop(192, 300),
            #         T.RandomResize(scales, max_size=640),
            # ])
                T.Compose([
                    T.RandomResize(scales, max_size=640),
                    T.RandomResize([320, 400, 480]),
                    T.RandomSizeCrop(308, 480),
                    T.RandomResize(scales, max_size=640),
            ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_hr_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transforms(image_set, args):
    type2transform = {
        'old_vistr': make_coco_transforms,
        'fixed_vistr': make_fixed_vistr_coco_transforms,
        'fixed_vistr_photo': make_fixed_vistr_photo_coco_transforms,
        'fixed_vistr_short': make_fixed_vistr_short_coco_transforms,
        'fixed_low_res_no_crop': make_fixed_low_res_no_crop_coco_transforms,
        'fixed_low_res': make_fixed_low_res_coco_transforms,
        'hr': make_hr_coco_transforms,
        'fixed_low_res_photo_no_crop': make_fixed_low_res_photo_no_crop_coco_transforms,
        'fixed_low_res_crop': make_fixed_low_res_crop_coco_transforms
    }
    print(f'Training with {args.aug_type} augmentation.')
    return type2transform[args.aug_type](image_set, args)