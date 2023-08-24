import numpy as np
import point_cloud_utils as pcu


def pts_nms(pts, preds, logits, affordance, threshold_confidence=0.5, threshold_dist=0.01):
        
        # (1) filter points via confidence threshold
        assert len(logits[0]) == 1
        if len(logits[0]) == 2:
            mask = logits[:, 1] >= threshold_confidence
        else:
            mask = logits[:, 0] >= threshold_confidence
        
        pts = pts[mask]
        preds = preds[mask]
        logits = logits[mask]
        affordance = affordance[mask].reshape(-1, 1)
        
        # (2) filter out negative points
        # mask = preds.reshape(-1) == 1
        # pts = pts[mask]
        # preds = preds[mask]
        # logits = logits[mask]
        
        if len(logits) <= 0:
            return None, None, None, None
        
        # filter points via distance
        if len(logits[0]) == 2:
            orders = np.argsort(logits[:, 1])   
        else: 
            orders = np.argsort(logits[:, 0])                
        new_pts = []         
        new_preds = []         
        new_logits = []
        new_affordance = []
        
        while(1):
            # find max_pt
            max_pt_idx = orders[-1]
            max_pt = pts[max_pt_idx]
            max_pred = preds[max_pt_idx]
            max_logit = logits[max_pt_idx]
            max_affordance = affordance[max_pt_idx]
            
            # keep max_pt
            new_pts.append(max_pt.reshape(1, 3))         
            new_preds.append(max_pred)       
            new_logits.append(max_logit)
            new_affordance.append(max_affordance)
            
            # compute distance
            dists = pcu.pairwise_distances(pts[orders], max_pt.reshape(1, 3))
            
            # remove neighbors
            mask = np.full((len(orders), ), True)
            mask[dists <= threshold_dist] = False
            orders = orders[mask] # remove max_pt and its neighborhood
            
            # no more points
            if len(orders) == 0:
                break
            
        new_pts = np.concatenate(new_pts)
        new_preds = np.concatenate(new_preds)
        new_logits = np.concatenate(new_logits)     
        new_affordance = np.concatenate(new_affordance)        
        return new_pts, new_preds, new_logits, new_affordance