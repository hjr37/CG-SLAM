import torch.nn.functional as F
import torch
from utils.loop_detection.extractors import netvlad

class LoopDetector():
    def __init__(self, conf, device) -> None:
        self.img_db = []
        self.detector = netvlad.NetVLAD(conf).eval().to(device)
        self.des_db = []
        self.loop_launch_id = None
        self.min_time_diff = None
        self.sim_threshold = None
    
    def get_frame_des(self, image):
        image = image.unsqueeze(0).type(torch.float32)
        image = torch.clamp(image,min=0,max=1)
        des = self.detector({'image':image})['global_descriptor']
        return des

    def add_des(self, des):
        # des = self.get_frame_des(frame)
        self.des_db.append(des)

    def keyframe_selection(self, cur_des, num):
        if num == 0:
            return []
        history_des = torch.cat(self.des_db[:-1], dim=0)
        sim_score = F.cosine_similarity(cur_des, history_des)
        _, ba_ids = torch.topk(sim_score, num)
        return ba_ids.tolist()


        
    
    def detection(self, cur_frame, keyframe_list):

        if len(self.des_db) < self.loop_launch_id:   # loop start from loop_launch_id
            return None

        cur_des  =self.get_frame_des(cur_frame)
        candidate_des_ls = torch.cat(self.des_db, dim=0)
        sim_score = F.cosine_similarity(cur_des, candidate_des_ls)

        max_score = torch.max(sim_score)
        match_frame_id = torch.argmax(sim_score)
        # return {'similiar_score':max_score, 'id':match_frame_id}

        if (cur_frame.id - keyframe_list[match_frame_id].id) < self.min_time_diff:  # time filter
            return None
        
        if max_score < self.sim_threshold: # score filter
            return None

        return {'similiar_score':max_score, 'id':match_frame_id}
    # @classmethod