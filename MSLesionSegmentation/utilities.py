from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import datetime
import random
import sklearn
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from Losses import AvPosDet, AvNegDet, FPR, FNR, TPR, TNR

def dice_metric(predicted, target):

    

    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1-dice_value(predicted, target).item()

    return value

def avertrue(predicted, target):

    sensloss = AvPosDet(to_onehot_y=True, sigmoid=True, squared_pred=True)
    senset = sensloss(predicted, target).item()
    return senset

def averfalse(predicted, target):

    sensloss = AvNegDet(to_onehot_y=True, sigmoid=True, squared_pred=True)
    sensef = sensloss(predicted, target).item()
    return sensef

def FPRate(predicted, target):

    ratefp = FPR(to_onehot_y=True, sigmoid=True, squared_pred=True)
    FPRa = ratefp(predicted, target).item()
    return FPRa

def FNRate(predicted, target):

    ratefn = FNR(to_onehot_y=True, sigmoid=True, squared_pred=True)
    FNRa = ratefn(predicted, target).item()
    return FNRa

def TPRate(predicted, target):

    ratetp = TPR(to_onehot_y=True, sigmoid=True, squared_pred=True)
    TPRa = ratetp(predicted, target).item()
    return TPRa

def TNRate(predicted, target):

    ratetn = TNR(to_onehot_y=True, sigmoid=True, squared_pred=True)
    TNRa = ratetn(predicted, target).item()
    return TNRa


def Dicefunc(predicted, target):

    ratefn = Dicenew(to_onehot_y=True, sigmoid=True, squared_pred=True)
    dicey = ratefn(predicted, target).item()
    return dicey





def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights` 
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda:0")):
    best_metric = -1
    AvtrueDice = -1
    AvfalseDice = 1
    tprDice = -1
    fprDice = 1
    tnrDice = -1
    fnrDice = 1
    best_single_dice = -1
    best_metric_tpr = 0
    best_single_tpr = 0
    best_metric_tnr = 0
    best_single_tnr = 0
    best_metric_Avtrue =0
    best_single_Avtrue = 0
    best_metric_fnr = 1
    best_single_fnr = 1
    best_metric_fpr = 1
    best_single_fpr = 1
    best_metric_Avfalse = 1
    best_single_Avfalse = 1
    best_metric_epoch = -1
    best_metric_epoch_tpr = -1
    best_metric_epoch_fpr = -1
    best_metric_epoch_fnr = -1
    best_metric_epoch_tnr = -1
    best_metric_epoch_Avfalse = -1
    best_metric_epoch_Avtrue = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_train_tpr = []
    save_metric_train_fpr = []
    save_metric_train_fnr = []
    save_metric_train_tnr = []
    save_metric_train_Avtrue = []
    save_metric_train_Avfalse = []
    save_metric_test = []
    save_metric_test_tpr = []
    save_metric_test_fpr = []
    save_metric_test_fnr = []
    save_metric_test_tnr = []
    save_metric_test_Avtrue = []
    save_metric_test_Avfalse = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step =0
        epoch_metric_train = 0
        tpr_metric_train = 0
        fpr_metric_train = 0
        fnr_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric= dice_metric(outputs, label)
            epoch_metric_train += train_metric
            #tpr_metric_train += tprtrain
            #fpr_metric_train += fprtrain
            #fnr_metric_train += fnrtrain

            print(f'Train_dice: {train_metric:.4f}')
            #print(f'True_positive_rate: {tprtrain:.4f}')
            #print(f'False_positive_rate: {fprtrain:.4f}')
            #print(f'False_negative_rate: {fnrtrain:.4f}')
        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        #training loss at particular epoch across all training images
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')
        #Dice  at particular epoch across all training images
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        #tpr_metric_train /= train_step
        #print(f'Epoch_metric_tpr: {tpr_metric_train:.4f}')
        #save_metric_train_tpr.append(tpr_metric_train)
        #np.save(os.path.join(model_dir, 'metric_train_tpr.npy'), save_metric_train_tpr)

        #fpr_metric_train /= train_step
        #print(f'Epoch_metric_fpr: {fpr_metric_train:.4f}')
        #save_metric_train_fpr.append(fpr_metric_train)
        #np.save(os.path.join(model_dir, 'metric_train_fpr.npy'), save_metric_train_fpr)

        #fnr_metric_train /= train_step
        #print(f'Epoch_metric_fnr: {fnr_metric_train:.4f}')
        #save_metric_train_fnr.append(fnr_metric_train)
        #np.save(os.path.join(model_dir, 'metric_train_fnr.npy'), save_metric_train_fnr)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                #test_metric = 0
                #tpr_test = 0
                #fpr_test = 0
                #fnr_test = 0
                epoch_metric_test = 0
                tpr_metric_test = 0
                fpr_metric_test = 0
                fnr_metric_test = 0
                tnr_metric_test = 0
                Avtrue_test = 0
                Avfalse_test =0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = test_label != 0
                    #print(torch.count_nonzero(test_label == True))
                    
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    tpr_test = TPRate(test_outputs, test_label)
                    tnr_test = TNRate(test_outputs, test_label)
                    fpr_test = FPRate(test_outputs, test_label)
                    fnr_test = FNRate(test_outputs, test_label)
                    AverageTrue =avertrue(test_outputs, test_label)
                    AverageFalse = averfalse(test_outputs, test_label)



                    #print(f'DICE: {test_metric:.4f}')
                    #print(f'truepositive: {tpr_test:.4f}')
                    #print(f'falsepositive: {fpr_test}')
                    #print(f'falsenegative: {fnr_test:.4f}')

#                    print(f'TP_test: {tprtest:.4f}')
 #                   print(f'FP_test: {fprtest:.4f}')
 #                   print(f'FN_test: {fnrtest:.4f}')



                    epoch_metric_test += test_metric
                    tpr_metric_test += tpr_test
                    fpr_metric_test += fpr_test
                    fnr_metric_test += fnr_test
                    tnr_metric_test += tnr_test
                    Avtrue_test += AverageTrue
                    Avfalse_test += AverageFalse

                #note that test_metric, tpr_test are the test values for a particular epoch whilst the best_metric values further down derived from epoch_metric_test, tpr_metric_test are used to find at what point the best mean values are produced, the best value of which indicating the point at which the algorithm is optimally trained
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
                #test loss in the final epoch across all testing images
                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)
                #Dice in the final epoch across all test images
                Avtrue_test /= test_step
                Avfalse_test /= test_step
                tpr_metric_test /= test_step
                fpr_metric_test /= test_step
                tnr_metric_test /= test_step
                fnr_metric_test /= test_step
                print(test_step)
                print(tpr_metric_test)
        

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    AvtrueDice = Avtrue_test
                    AvfalseDice = Avfalse_test
                    tprDice = tpr_metric_test
                    fprDice = fpr_metric_test
                    tnrDice = tnr_metric_test
                    fnrDice = fnr_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))

                



#               print(tp_test)
#               print(fn_test)
#               print(tpr_test)
#               print(tpr_metric_test)

                print(f'test_Avtrue_epoch: {Avtrue_test:.4f}')
                save_metric_test_Avtrue.append(Avtrue_test)
                np.save(os.path.join(model_dir, 'metric_test_Avtrue.npy'), save_metric_test_Avtrue)
                if Avtrue_test > best_metric_Avtrue:
                    best_metric_Avtrue = Avtrue_test
                    best_metric_epoch_Avtrue = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "Avtrue_best_metric_model.pth"))

                



                print(f'test_Avfalse_epoch: {Avfalse_test:.4f}')
                save_metric_test_Avfalse.append(Avfalse_test)
                np.save(os.path.join(model_dir, 'metric_test_Avfalse.npy'), save_metric_test_Avfalse)
                if Avfalse_test < best_metric_Avfalse:
                    best_metric_Avfalse = Avfalse_test
                    best_metric_epoch_Avfalse = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "Avfalse_best_metric_model.pth"))
                    

                        
 
                print(f'test_tpr_epoch: {tpr_metric_test:.4f}')
                save_metric_test_tpr.append(tpr_metric_test)
                np.save(os.path.join(model_dir, 'metric_test_tpr.npy'), save_metric_test_tpr)
                if tpr_metric_test > best_metric_tpr:
                    best_metric_tpr = tpr_metric_test
                    best_metric_epoch_tpr = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "tpr_best_metric_model.pth"))

#               print(fp_test)
#               print(tn_test)
#               print(fpr_test)
#               print(fpr_metric_test)

                print(f'test_fpr_epoch: {fpr_metric_test:.4f}')
                save_metric_test_fpr.append(fpr_metric_test)
                np.save(os.path.join(model_dir, 'metric_test_fpr.npy'), save_metric_test_fpr)
                if fpr_metric_test < best_metric_fpr:
                    best_metric_fpr = fpr_metric_test
                    best_metric_epoch_fpr = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "fpr_best_metric_model.pth"))
                        


                print(f'test_fnr_epoch: {fnr_metric_test:.4f}')
                save_metric_test_fnr.append(fnr_metric_test)
                np.save(os.path.join(model_dir, 'metric_test_fnr.npy'), save_metric_test_fnr)
                if  fnr_metric_test < best_metric_fnr:
                    best_metric_fnr = fnr_metric_test 
                    best_metric_epoch_fnr = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "fnr_best_metric_model.pth"))
                

                

                print(f'test_tnr_epoch: {tnr_metric_test:.4f}')
                save_metric_test_tnr.append(tnr_metric_test)
                np.save(os.path.join(model_dir, 'metric_test_tnr.npy'), save_metric_test_tnr)
                if  tnr_metric_test > best_metric_tnr:
                    best_metric_tnr = tnr_metric_test 
                    best_metric_epoch_tnr = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "tnr_best_metric_model.pth"))
                

                
                print(
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                print(
                    f"\nbest mean AverageTrueRate: {best_metric_Avtrue:.4f} "
                    f"at epoch: {best_metric_epoch_Avtrue}"
                )
                print(

                    f"\nbest mean AverageFalseRate: {best_metric_Avfalse:.4f} "
                    f"at epoch: {best_metric_epoch_Avfalse}"
                )
                print(

                    f"\nbest mean tpr: {best_metric_tpr:.4f} "
                    f"at epoch: {best_metric_epoch_tpr}"
                )
                print(
                    f"\nbest mean fpr: {best_metric_fpr:.4f} "
                    f"at epoch: {best_metric_epoch_fpr}"
                )
                print(
                    f"\nbest mean fnr: {best_metric_fnr:.4f} "
                    f"at epoch: {best_metric_epoch_fnr}"
                )
                print(
                    f"\nbest mean tnr: {best_metric_tnr:.4f} "
                    f"at epoch: {best_metric_epoch_tnr}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch} \n"
        f"train completed, Avtrue at best Dice: {AvtrueDice:.4f} \n"
        f"train completed, Avfalse at best Dice: {AvfalseDice:.4f} \n"
        f"train completed, fpr at best Dice: {fprDice:.4f}\n "
        f"train completed, tpr at best Dice: {tprDice:.4f} \n"
        f"train completed, tnr at best Dice: {tnrDice:.4f} \n"
        f"train completed, fnr at best Dice: {fnrDice:.4f} \n"
        f"all at epoch: {best_metric_epoch} \n"
        f"train completed, best_metric_AVtrue: {best_metric_Avtrue:.4f} "
        f"at epoch: {best_metric_epoch_Avtrue} \n"
        f"train completed, best_metric_AVfalse: {best_metric_Avfalse:.4f} "
        f"at epoch: {best_metric_epoch_Avfalse} \n"
        f"train completed, best_metric_tpr: {best_metric_tpr:.4f} "
        f"at epoch: {best_metric_epoch_tpr} \n"
        f"train completed, best_metric_fpr: {best_metric_fpr:.4f} "
        f"at epoch: {best_metric_epoch_fpr} \n"
        f"train completed, best_metric_fnr: {best_metric_fnr:.4f} "
        f"at epoch: {best_metric_epoch_fnr} \n"
        f"train completed, best_metric_tnr: {best_metric_tnr:.4f} "
        f"at epoch: {best_metric_epoch_tnr} \n")




def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that you can si if the it is okay or you need 
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()


def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val