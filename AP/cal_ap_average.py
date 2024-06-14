import os
import csv

mode = "val"
class_type = ["car"]
detection_type = ["_detection_3d"]

attack_type = ["kitti", "camera_laser_hiding", "camera_projection_creating", "camera_laser_strip_injection", "camera_emi_strip_loss", "camera_emi_truncation","camera_acoustic_blur_linear",\
"lidar_laser_hiding", "lidar_laser_creating_car", "lidar_laser_arbitrary_point_injection", "lidar_laser_background_noise_injection","lidar_emi_gaussian_noise"]
# model_type = ["VirConv-T"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/VirConv/output/kitti/VirConv-T/default/eval/epoch_2/val/"
# model_type = ["VirConv-L"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/VirConv/output/kitti/VirConv-L/default/eval/epoch_2/val/"
# model_type = ["PP"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools/cfgs/kitti_models/pointpillar_painted/default/eval/epoch_80/val/"
# model_type = ["CLOCs"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/CLOCs/second/results/"
# model_type = ["EPNet"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/EPNet/tools/log/Car/models/full_epnet_with_iou_branch/eval_results/"
model_type = ["AVOD"]
folders_path = "/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/"
# model_type = ["FP"]
# folders_path = "/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/"
csv_name = "AP_" + mode + "_average_" + model_type[0] + ".csv"
result_csv = open(csv_name,'w')
writer = csv.writer(result_csv)
header = ['model', 'attack','ap', 'ap_delta','AP_relative','fp', 'fp_delta','fp_relative','fn', 'fn_delta','fn_relative','fnr','fnr_delta','fpr', 'fpr_delta']
writer.writerow(header)
for model in model_type:
    for attack in attack_type:
        # ap_dir = folders_path + attack + "/final_result"
        # ap_dir = folders_path + attack + "/eval/epoch_46/val/final_result" # EPNet
        # ap_dir = folders_path + attack # CLOCs
        ap_dir = folders_path + attack + "/predictions/kitti_native_eval/0.1/120000" # AVOD
        # ap_dir = folders_path + f"detection_results_v1_{attack}" # FP
        if os.path.exists(ap_dir):
            info_file = ap_dir + "/" + "stats_car_detection_3d.txt"
            f_info = open(info_file, "r")
            groups = f_info.read().split("\n\n")
            threshold_easy = float(groups[0].split("\n")[-1].split(" ")[1])
            tp_easy = float(groups[0].split("\n")[-1].split(" ")[3])
            fp_easy = float(groups[0].split("\n")[-1].split(" ")[5])
            fn_easy = float(groups[0].split("\n")[-1].split(" ")[7])
            fnr_easy = int(fn_easy)/(int(fn_easy)+int(tp_easy))
            fpr_easy = int(fp_easy)/(int(fp_easy)+int(tp_easy))

            threshold_moderate = float(groups[1].split("\n")[-1].split(" ")[1])
            tp_moderate = float(groups[1].split("\n")[-1].split(" ")[3])
            fp_moderate = float(groups[1].split("\n")[-1].split(" ")[5])
            fn_moderate = float(groups[1].split("\n")[-1].split(" ")[7])
            fnr_moderate = int(fn_moderate)/(int(fn_moderate)+int(tp_moderate))
            fpr_moderate = int(fp_moderate)/(int(fp_moderate)+int(tp_moderate))

            threshold_hard = float(groups[2].split("\n")[-1].split(" ")[1])
            tp_hard = float(groups[2].split("\n")[-1].split(" ")[3])
            fp_hard = float(groups[2].split("\n")[-1].split(" ")[5])
            fn_hard = float(groups[2].split("\n")[-1].split(" ")[7])
            fnr_hard = int(fn_hard)/(int(fn_hard)+int(tp_hard))
            fpr_hard = int(fp_hard)/(int(fp_hard)+int(tp_hard))

            AP_easy = float(groups[3].split(" ")[-3])
            AP_moderate = float(groups[3].split(" ")[-2])
            AP_hard = float(groups[3].split(" ")[-1].split("\n")[0])
            AP_average = (AP_easy + AP_moderate + AP_hard) / 3
            fp_average = (fp_easy + fp_moderate + fp_hard) / 3
            fn_average = (fn_easy + fn_moderate + fn_hard) / 3
            fnr_average = (fnr_easy + fnr_moderate + fnr_hard) / 3
            fpr_average = (fpr_easy + fpr_moderate + fpr_hard) / 3
        if attack == 'kitti':
            kitti_data = [AP_average, fp_average,fn_average,fnr_average,fpr_average]
            content = [model, attack,AP_average,0,0,fp_average,0,0,fn_average,0,0,fnr_average,0,fpr_average,0]
            
        else:
            content = [model, attack,AP_average,AP_average-kitti_data[0],(AP_average-kitti_data[0])/kitti_data[0],fp_average,fp_average-kitti_data[1],(fp_average-kitti_data[1])/kitti_data[1],fn_average,fn_average-kitti_data[2],(fn_average-kitti_data[2])/kitti_data[2],fnr_average,fnr_average-kitti_data[3],fpr_average,fpr_average-kitti_data[4]]
            
        writer.writerow(content)
result_csv.close()

