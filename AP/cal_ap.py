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
# model_type = ["CLOCs"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/CLOCs/second/results/"
# model_type = ["EPNet"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/EPNet/tools/log/Car/models/full_epnet_with_iou_branch/eval_results/"
# model_type = ["PP"]
# folders_path = "/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools/cfgs/kitti_models/pointpillar_painted/default/eval/epoch_80/val/"

model_type = ["AVOD"]
folders_path = "/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/"
# model_type = ["FP"]
# folders_path = "/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/"

csv_name = "AP_" + mode + "_" + model_type[0] + ".csv"
result_csv = open(csv_name,'w')
writer = csv.writer(result_csv)
header = ['model', 'attack','difficulty', 'threshold','ap', 'ap_delta','AP_relative','fp', 'fp_delta','fp_relative','fn', 'fn_delta','fn_relative','fnr','fnr_delta','fpr', 'fpr_delta']
writer.writerow(header)
for model in model_type:
    for attack in attack_type:
        # ap_dir = folders_path + attack + "/final_result"
        # ap_dir = folders_path + attack + "/eval/epoch_46/val/final_result" # EPNet
        # ap_dir = folders_path + attack + "/final_result" # CLOCs
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
        if attack == 'kitti':
            kitti_data = [AP_easy, AP_moderate, AP_hard, threshold_easy, threshold_moderate, threshold_hard, fp_easy, fp_moderate, fp_hard, fn_easy, fn_moderate, fn_hard, fnr_easy, fnr_moderate, fnr_hard, fpr_easy, fpr_moderate, fpr_hard]
            content1 = [model, attack, 'easy',threshold_easy,AP_easy,0,0,fp_easy,0,0,fn_easy,0,0,fnr_easy,0,fpr_easy,0]
            content2 = [model, attack, 'moderate', threshold_moderate,AP_moderate,0,0,fp_moderate,0,0,fn_moderate,0,0,fnr_moderate,0,fpr_moderate,0]         
            content3 = [model, attack, 'hard',threshold_hard,AP_hard,0,0,fp_hard,0,0,fn_hard,0,0,fnr_hard,0,fpr_hard,0]
        else:
            content1 = [model, attack, 'easy', threshold_easy,AP_easy,AP_easy-kitti_data[0],(AP_easy-kitti_data[0])/kitti_data[0],fp_easy,fp_easy-kitti_data[6],(fp_easy-kitti_data[6])/kitti_data[6],fn_easy,fn_easy-kitti_data[9],(fn_easy-kitti_data[9])/kitti_data[9],fnr_easy,fnr_easy-kitti_data[12],fpr_easy,fpr_easy-kitti_data[15]]
            content2 = [model, attack, 'moderate',threshold_moderate,AP_moderate,AP_moderate-kitti_data[1],(AP_moderate-kitti_data[1])/kitti_data[1],fp_moderate,fp_moderate-kitti_data[7],(fp_moderate-kitti_data[7])/kitti_data[7],fn_moderate,fn_moderate-kitti_data[10],(fn_moderate-kitti_data[10])/kitti_data[10],fnr_moderate,fnr_moderate-kitti_data[13],fpr_moderate,fpr_moderate-kitti_data[16]]
            content3 = [model, attack, 'hard',threshold_hard,AP_hard,AP_hard-kitti_data[2],(AP_hard-kitti_data[2])/kitti_data[2],fp_hard,fp_hard-kitti_data[8],(fp_hard-kitti_data[8])/kitti_data[8],fn_hard,fn_hard-kitti_data[11],(fn_hard-kitti_data[11])/kitti_data[11],fnr_hard,fnr_hard-kitti_data[14],fpr_hard,fpr_hard-kitti_data[17]]
            
        writer.writerow(content1)
        writer.writerow(content2)
        writer.writerow(content3)
result_csv.close()
