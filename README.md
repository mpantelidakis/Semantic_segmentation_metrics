# Semantic_segmentation_metrics

### Generating confusion matrices and evaluation metrics
First, modify the code in gen_conf.py and add the directories for which you want to generate metrics.

An example directory would contain ground truths and predictions with the following naming convention xxx_gt.png for a ground truth, and xxx_pred.png for a prediction.

Then, change the resolution to match the one of your images in gen_conf.py.

### Results

After running the script, you will notice that a new directory "conf_mat_results/" was created.

In that directory, you will find the confusion matrices for all your models, as well as a .csv file with performance metrics.

The supported metrics are accuracy, precision, recall, F1-score, and mean IoU.
