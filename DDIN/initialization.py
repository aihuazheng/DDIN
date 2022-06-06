from torchvision import transforms

def init_crossx_params(backbone, datasetname):

   
    gamma1, gamma2, gamma3 = 0.0, 0.0, 0.0
    lr = 0.0

    if backbone is 'senet':
        if datasetname is 'nabirds':
            gamma1 = 0.1
            gamma2 = 0.25
            gamma3 = 0.5
        elif datasetname in ['cubbirds', 'stcars']:
            gamma1 = 1
            gamma2 = 0.25
            gamma3 = 1
        elif datasetname is 'stdogs':
            gamma1 = 1
            gamma2 = 0.5
            gamma3 = 1
        elif datasetname is 'vggaricraft':
            gamma1 = 0.5
            gamma2 = 0.1
            gamma3 = 0.1
        else:
            pass
    elif backbone is 'resnet':
        if datasetname in ['nabirds', 'cubbirds']:
            gamma1 = 0.5
            gamma2 = 0.25
            gamma3 = 0.5
        elif datasetname is 'stcars':
            gamma1 = 1
            gamma2 = 0.25
            gamma3 = 1
        elif datasetname is 'stdogs':
            gamma1 = 0.01
            gamma2 = 0.01
            gamma3 = 1
        elif datasetname is 'vggaricraft':
            gamma1 = 0.5
            gamma2 = 0.1
            gamma3 = 0.5
        else:
            pass
    else:
        pass
    
    if datasetname is 'stdogs':
        lr = 0.001
    else:
        lr = 0.01

    return gamma1, gamma2, gamma3, lr



if __name__ == "__main__":
    pass