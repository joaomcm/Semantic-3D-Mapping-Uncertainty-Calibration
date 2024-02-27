import numpy as np
import pdb
import torch.nn as nn
import torch

class Calibration_calc:
    def __init__(self, tiers =np.arange(11)/10,no_void = False,one_hot = False):
        self.tiers = tiers
        self.total_bin_members = np.zeros(len(tiers)-1)
        self.correct_bin_members = np.zeros(len(tiers)-1)
        self.total_bin_confidence = np.zeros(len(tiers)-1)
        self.no_void = no_void
        self.one_hot = one_hot
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.no_void):
            if(self.one_hot):
                gt_labels = semantic_label_gt.argmax(axis=1)
            else:
                gt_labels = semantic_label_gt
            semantic_label_gt = semantic_label_gt[gt_labels != 0]
            semantic_label = semantic_label[gt_labels!=0]
        max_conf = semantic_label.max(axis = 1)
        # total_bin_members = np.zeros(len(self.tiers)-1)
        # correct_bin_members = np.zeros(len(self.tiers)-1)
        pred = semantic_label.argmax(axis =1)
        comparison_sheet = semantic_label_gt == pred
        for i in range(len(self.tiers)-1):
            pdb.set_trace()
            if(self.tiers[i+1] != 1.0):
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<self.tiers[i+1])
            else:
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<=self.tiers[i+1])
            self.total_bin_members[i] += conf_mask_tier.sum()
            self.correct_bin_members[i] += comparison_sheet[conf_mask_tier].sum()
            self.total_bin_confidence[i] += max_conf[conf_mask_tier].sum()
    def return_calibration_results(self):
        return self.correct_bin_members/self.total_bin_members,self.total_bin_confidence/self.total_bin_members,self.tiers[1:]

    def get_ECE(self):
        acc = self.correct_bin_members/self.total_bin_members
        conf = self.total_bin_confidence/self.total_bin_members
        share = np.nan_to_num(((self.total_bin_members)/(self.total_bin_members.sum())),nan=0)
        return (share*np.nan_to_num(np.abs(acc-conf),nan = 0)).sum()

class Calibration_calc_3D:
    def __init__(self, tiers =np.arange(11)/10,no_void = False,one_hot = True):
        self.tiers = tiers
        self.total_bin_members = np.zeros(len(tiers)-1)
        self.correct_bin_members = np.zeros(len(tiers)-1)
        self.total_bin_confidence = np.zeros(len(tiers)-1)
        self.no_void = no_void
        self.one_hot = one_hot
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.no_void):
            if(self.one_hot):
                gt_labels = semantic_label_gt.argmax(axis=1)
            else:
                gt_labels = semantic_label_gt
            semantic_label_gt = semantic_label_gt[gt_labels != 0]
            semantic_label = semantic_label[gt_labels!=0]
        max_conf = semantic_label.max(axis = 1)
        # total_bin_members = np.zeros(len(self.tiers)-1)
        # correct_bin_members = np.zeros(len(self.tiers)-1)
        pred = semantic_label.argmax(axis =1)
        if(self.one_hot):
            comparison_sheet = semantic_label_gt.argmax(axis=1) == pred
        else:
            comparison_sheet = semantic_label_gt == pred
        for i in range(len(self.tiers)-1):
            if(self.tiers[i+1] != 1.0):
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<self.tiers[i+1])
            else:
                conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<=self.tiers[i+1])
#             pdb.set_trace()
            self.total_bin_members[i] += conf_mask_tier.sum()
            self.correct_bin_members[i] += comparison_sheet[conf_mask_tier].sum()
            self.total_bin_confidence[i] += max_conf[conf_mask_tier].sum()
    def return_calibration_results(self):
        return self.correct_bin_members/self.total_bin_members,self.total_bin_confidence/self.total_bin_members,self.tiers[1:]
    
    def get_ECE(self):
        if(np.all(self.total_bin_members == 0)):
            return np.nan
        else:
            acc = self.correct_bin_members/self.total_bin_members
            conf = self.total_bin_confidence/self.total_bin_members
            
            share = np.nan_to_num(((self.total_bin_members)/(self.total_bin_members.sum())),nan=0)
            # print(share,np.abs(acc-conf))
            return (share*np.nan_to_num(np.abs(acc-conf),nan = 0)).sum()

class mECE_Calibration_calc_3D:
    def __init__(self, tiers =np.arange(11)/10,no_void = False,one_hot = True,n_classes = 21):
        self.tiers = tiers
        self.no_void = no_void
        self.one_hot = one_hot
        self.n_classes = n_classes
        self.cals = {}
        self.agg_cal = Calibration_calc_3D(tiers = self.tiers,no_void = self.no_void,one_hot = self.one_hot)
        for i in range(self.n_classes):
            self.cals.update({i:Calibration_calc_3D(self.tiers,self.no_void,self.one_hot)})
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.one_hot):
            map_gt = semantic_label_gt.argmax(axis = 1)
        else:
            map_gt = semantic_label_gt
        self.agg_cal.update_bins(semantic_label,semantic_label_gt)
        for i in range(self.n_classes):
            mask = map_gt == i
            if(map_gt[mask].shape[0]>0):
                self.cals[i].update_bins(semantic_label[mask],semantic_label_gt[mask])
    def return_calibration_results(self):
        results = {}
        for i in range(self.n_classes):
            results.update({i:self.cals[i].return_calibration_results()})
        results.update({'aggregate':self.agg_cal.return_calibration_results()})
        return results
    def get_ECEs(self):
        results = []
        for i in range(self.n_classes):
            results.append(self.cals[i].get_ECE())
        results.append(self.agg_cal.get_ECE())
        # results.append(self.get_TL_ECE())
        return results 
    def get_mECE(self):
        ECEs = []
        for i in range(self.n_classes):
            if(i !=0):
                ECEs.append(self.cals[i].get_ECE())
        ECEs = np.array(ECEs)
        #filtering out pesky nans due to bad calibrations that end up with no predictions in the fixed case and penalizing those cases
        ECEs[np.logical_not(np.isfinite(ECEs))] = 1.0
        return np.mean(ECEs)
    def get_TL_ECE(self):
        accuracies = []
        confidences = []
        memberships = []
        for i in range(self.n_classes):
            acc,conf,borders = self.cals[i].return_calibration_results()
            membership = self.cals[i].total_bin_members
            accuracies.append(acc)
            confidences.append(conf)
            memberships.append(membership)
        accuracies = np.array(accuracies)
        confidences = np.array(confidences)
        memberships = np.array(memberships)
        bin_membership_totals = memberships.sum(axis =0,keepdims = True)
        within_bin_fractions = np.nan_to_num(memberships/bin_membership_totals,nan = 0,posinf = 0,neginf = 0)
        differences = np.nan_to_num(np.abs(accuracies-confidences),nan = 0,posinf = 0,neginf = 0)
        mean_bin_differences = (differences*within_bin_fractions).sum(axis = 0)
        bin_fractions = bin_membership_totals/bin_membership_totals.sum()
        weighted_delta_bs = np.nan_to_num(bin_fractions*mean_bin_differences,nan = 0,posinf = 0,neginf = 0)
        TL_ECE = weighted_delta_bs.sum()
        return TL_ECE


class All_Predictions_Calibration:
    def __init__(self, tiers =np.arange(11)/10):
        self.tiers = tiers
        self.total_bin_members = np.zeros(len(tiers)-1)
        self.correct_bin_members = np.zeros(len(tiers)-1)
        self.total_bin_confidence = np.zeros(len(tiers)-1)

    def update_bins(self,semantic_label,semantic_label_gt):
        for j in range(semantic_label.shape[2]):
            max_conf = semantic_label[:,:,j]
            # total_bin_members = np.zeros(len(self.tiers)-1)
            # correct_bin_members = np.zeros(len(self.tiers)-1)
            pred = semantic_label
            comparison_sheet = semantic_label_gt == j
            for i in range(len(self.tiers)-1):
                if(self.tiers[i+1] != 1.0):
                    conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<self.tiers[i+1])
                else:
                    conf_mask_tier = np.logical_and(max_conf>=self.tiers[i],max_conf<=self.tiers[i+1])
                self.total_bin_members[i] += conf_mask_tier.sum()
                self.correct_bin_members[i] += comparison_sheet[conf_mask_tier].sum()
                self.total_bin_confidence[i] += max_conf[conf_mask_tier].sum()

    def return_calibration_results(self):
        return self.correct_bin_members/self.total_bin_members,self.total_bin_confidence/self.total_bin_members,self.tiers[1:]

    def get_ECE(self):
        acc = self.correct_bin_members/self.total_bin_members
        conf = self.total_bin_confidence/self.total_bin_members
        share = np.nan_to_num(((self.total_bin_members)/(self.total_bin_members.sum())),nan=0)
        return (share*np.nan_to_num(np.abs(acc-conf),nan = 0)).sum()
    
class BrierScore3D:
    def __init__(self,n_classes = 21,no_void = True,one_hot = False):
        self.n_classes = n_classes
        self.no_void = no_void
        self.one_hot = one_hot
        self.total_entries = 0
        self.current_score = 0
    def update_bins(self,semantic_label,semantic_label_gt):
        if self.no_void:
            if(not self.one_hot):
                semantic_label_gt = nn.functional.one_hot(torch.from_numpy(semantic_label_gt.astype(np.int64)),num_classes = self.n_classes).numpy().astype(np.float32)

            no_void_mask = semantic_label_gt[:,0] != 1
            semantic_label = semantic_label[no_void_mask]
            semantic_label_gt = semantic_label_gt[no_void_mask]
            # pdb.set_trace()
            discrepancy = np.power(semantic_label-semantic_label_gt,2).sum(axis = 1).mean()
            entries = semantic_label.shape[0]
            # pdb.set_trace()
            self.current_score = (self.current_score*self.total_entries + discrepancy*entries)/(entries+self.total_entries)
            self.total_entries += entries
    def return_score(self):
        return self.current_score

class mECE_Calibration_calc_3D_fix(mECE_Calibration_calc_3D):
    def update_bins(self,semantic_label,semantic_label_gt):
        if(self.one_hot):
            map_gt = semantic_label_gt.argmax(axis = 1)
        else:
            map_gt = semantic_label_gt
        self.agg_cal.update_bins(semantic_label,semantic_label_gt)
        pred = semantic_label.argmax(axis =1)
        for i in range(self.n_classes):
            mask = pred == i
            if(pred[mask].shape[0]>0):
                self.cals[i].update_bins(semantic_label[mask],semantic_label_gt[mask])


class Cumulative_mIoU:
    def __init__(self,n_classes):
        self.n_classes = n_classes
        self.intersections = np.zeros(self.n_classes)
        self.unions = np.zeros(self.n_classes)
    def update_counts(self,pred,gt):
        for i in range(self.n_classes):
            gt_mask = gt == i
            pred_mask = pred == i
            self.intersections[i] += np.logical_and(gt_mask,pred_mask).sum()
            self.unions[i] += np.logical_or(gt_mask,pred_mask).sum()
    def get_IoUs(self):
        return self.intersections/self.unions
    
class Cumulative_mIoU_torch:
    def __init__(self,n_classes):
        self.n_classes = n_classes
        self.intersections = torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
        self.unions = torch.from_numpy(np.zeros(self.n_classes)).long().to('cuda:0')
    def update_counts(self,pred,gt):
        with torch.no_grad():
            for i in range(self.n_classes):
                gt_mask = torch.eq(gt.long(),i)
                pred_mask = torch.eq(pred.long(),i)
                self.intersections[i] += torch.logical_and(gt_mask,pred_mask).sum()
                self.unions[i] += torch.logical_or(gt_mask,pred_mask).sum()
    def get_IoUs(self):
        return self.intersections.cpu().numpy()/self.unions.cpu().numpy()

