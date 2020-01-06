# from dataloaders.datasets import combine_dbs, pascal, sbd
from dataloaders import custom_transforms as tr
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    # if args.dataset == 'pascal':
    # train_set = pascal.VOCSegmentation(args, split='train')
    # val_set = pascal.VOCSegmentation(args, split='val')
    # if args.use_sbd:
    #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
    #
    # num_class = train_set.NUM_CLASSES
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # TODO: add a crop here, because the images are not of the same size (Square or whatever)
    tfs = transforms.Compose([
        tr.TestMode(),
        tr.FixedResize(512),
        tr.RandomHorizontalFlip(),
        tr.RandomGaussianBlur(),
        tr.ToTensor()])
    data = ImageFolder(root=args.test_root, transform=tfs)
    test_loader = DataLoader(data, batch_size=args.batch_size, num_workers=0)

    return None, None, test_loader, 1

    # elif args.dataset == 'cityscapes':
    #     train_set = cityscapes.CityscapesSegmentation(args, split='train')
    #     val_set = cityscapes.CityscapesSegmentation(args, split='val')
    #     test_set = cityscapes.CityscapesSegmentation(args, split='test')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #
    #     return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class

