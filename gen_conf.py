from calendar import firstweekday
import os,cv2, sys
import numpy as np

import results_utils 
import helpers

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    

def plot_conf(total_tp, total_fn, total_fp, total_tn):
    import seaborn as sns
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 12})

    #Generate the confusion matrix
    cf_matrix = np.array([[total_tp, total_fn],
                        [total_fp, total_tn]])

    group_counts = [human_format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1} px ({v2})' for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    # ax.set_title(f'{dir.split("_")[1]} confusion matrix\n')
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual ')
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Sunlit leaf','Non-sunlit leaf'])
    ax.yaxis.set_ticklabels(['Sunlit leaf','Non-sunlit leaf'])

    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("center") 


    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig(f'conf_mat_results/{dir.split("_")[1]}conf.pdf')
    plt.show()

# Generates confusion matrices from gt and prediction files in given directories
dirs = []
GMM_dir= "Test_GMM"
FRRN_dir = "Test_FRRN"
Densenet_dir = "Test_Densenet"
DLV3_dir= "Test_DLV3"

dirs.append(GMM_dir)
dirs.append(FRRN_dir)
dirs.append(Densenet_dir)
dirs.append(DLV3_dir)

crop_height = 320
crop_width = 480
# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info("./class_dict.csv")
# print(class_names_list, label_values)
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Create results directory if needed
if not os.path.isdir("%s"%("conf_mat_results")):
        os.makedirs("%s"%("conf_mat_results"))

combined_metrics=open('conf_mat_results/combined_metrics.csv','w')
combined_metrics.write("model, accuracy, precision, recall, f1, mean IoU\n")

for dir in dirs:
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    gt_images, pred_images = results_utils.prepare_data(dir)
    print(f"\n\n{dir} contains {len(gt_images)} ground truth images and {len(pred_images)} predictions.")
    print("-----------------------------------------------------------------------------------------")

    target=open("conf_mat_results/%s.csv"%(dir),'w')
    target.write("file, TP, TN, FP, FN\n")

    # Run testing on ALL test images
    for ind in range(len(gt_images)):
        sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(gt_images)))
        sys.stdout.flush()

        pred = np.expand_dims(np.float32(results_utils.load_image(pred_images[ind])[:crop_height, :crop_width]),axis=0)/255.0
        gt = results_utils.load_image(gt_images[ind])[:crop_height, :crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        pred = np.array(pred[0,:,:,:])
        pred = helpers.reverse_one_hot(pred)
        out_vis_image = helpers.colour_code_segmentation(pred, label_values)

        tp, tn, fp, fn = results_utils.evaluate_segmentation(pred=pred, label=gt)

        total_tp+=tp
        total_tn+=tn
        total_fp+=fp
        total_fn+=fn
   
        file_name = results_utils.filepath_to_name(gt_images[ind]).replace('_gt','')
        target.write("%s, %f, %f, %f, %f"%(file_name, tp, tn, fp, fn))
        target.write("\n")

    target.close()





    print("\n\nTP: ", total_tp)
    print("\nFP: ", total_fp)
    print("\nFN: ", total_fn)
    print("\nTN: ", total_tn)

    acc, prec, recall, f1, mean_iou, = results_utils.calc_metrics_from_conf(total_tp, total_fp, total_fn, total_tn)
    combined_metrics.write("%s, %f, %f, %f, %f, %f"%(dir.split('_')[1], acc, prec, recall, f1, mean_iou))
    combined_metrics.write("\n")

    plot_conf(total_tp, total_fn, total_fp, total_tn)
    
combined_metrics.close()
    




