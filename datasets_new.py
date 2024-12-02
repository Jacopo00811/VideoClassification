from glob import glob
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import torch.nn.functional as F 

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir='/ucf101_noleakage',
        split='train', 
        transform=None
    ): 
        self.base_dir = root_dir
        metadata_file = os.path.join(self.base_dir, "metadata", f"{split}.csv")
        self.frame_paths = sorted(glob(os.path.join(self.base_dir, "frames", split, "*", "*", "*.jpg")))
        self.df = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        matches = self.df.loc[self.df[attr] == value]
        if len(matches) == 0:
            raise ValueError(f"No metadata found for {attr}={value}")
        return {'label': matches.iloc[0]['label']}

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = os.path.basename(os.path.dirname(frame_path))
        
        try:
            video_meta = self._get_meta('video_name', video_name)
            label = video_meta['label']
            
            frame = Image.open(frame_path).convert("RGB")

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = T.ToTensor()(frame)

            return frame, label
        except Exception as e:
            print(f"Error processing frame {frame_path}")
            print(f"Video name extracted: {video_name}")
            print(f"Available video names:", self.df['video_name'].unique()[:5])
            raise

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir='/ucf101_noleakage',
        split='train', 
        transform=None,
        stack_frames=True
    ):
        self.base_dir = os.path.dirname(root_dir)  # Go up one level since root_dir is frames dir
        self.frames_dir = root_dir
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

        video_dir = os.path.join(self.base_dir, "videos")
        metadata_file = os.path.join(self.base_dir, "metadata", f"{split}.csv")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        self.video_paths = sorted(glob(os.path.join(video_dir, split, "*", "*.avi")))
        self.df = pd.read_csv(metadata_file)

        print(f"Loaded {len(self.video_paths)} videos and {len(self.df)} metadata entries")
        
    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        matches = self.df.loc[self.df[attr] == value]
        print(f"Looking for {attr}={value}")
        print(f"Found {len(matches)} matches")
        if len(matches) > 0:
            print(f"First match: {matches.iloc[0].to_dict()}")
            return {'label': matches.iloc[0]['label']}
        raise ValueError(f"No metadata found for {attr}={value}")

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).replace('.avi', '')
        
        try:
            video_meta = self._get_meta('video_name', video_name)
            label = video_meta['label']
            
            frames_dir = video_path.replace('videos', 'frames').replace('.avi', '')
            video_frames = self.load_frames(frames_dir)

            if self.transform:
                frames = [self.transform(frame) for frame in video_frames]
            else:
                frames = [T.ToTensor()(frame) for frame in video_frames]
            
            if self.stack_frames:
                frames = torch.stack(frames).permute(1, 0, 2, 3)

            return frames, label
            
        except Exception as e:
            print(f"Error processing video {video_name}")
            print(f"Video path: {video_path}")
            print(f"DataFrame matches:", self.df[self.df['video_name'] == video_name])
            raise
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames

class FlowVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir='/ucf101_noleakage',
        split='train', 
        resize=(64,64)
    ):
        self.base_dir = os.path.dirname(root_dir)  # Go up one level since root_dir is flows dir
        self.flows_dir = root_dir
        self.split = split
        self.resize = resize    
        self.n_sampled_frames = 10

        video_dir = os.path.join(self.base_dir, "videos")
        metadata_file = os.path.join(self.base_dir, "metadata", f"{split}.csv")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        self.video_paths = sorted(glob(os.path.join(video_dir, split, "*", "*.avi")))
        self.df = pd.read_csv(metadata_file)

        print(f"Loaded {len(self.video_paths)} videos and {len(self.df)} metadata entries")

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        matches = self.df.loc[self.df[attr] == value]
        print(f"Looking for {attr}={value}")
        print(f"Found {len(matches)} matches")
        if len(matches) > 0:
            print(f"First match: {matches.iloc[0].to_dict()}")
            return {'label': matches.iloc[0]['label']}
        raise ValueError(f"No metadata found for {attr}={value}")

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).replace('.avi', '')
        
        try:
            video_meta = self._get_meta('video_name', video_name)
            label = video_meta['label']
            
            flows_dir = video_path.replace('videos', 'flows').replace('.avi', '')
            flows = self.load_flows(flows_dir)

            return flows, label
            
        except Exception as e:
            print(f"Error processing video {video_name}")
            print(f"Video path: {video_path}")
            print(f"DataFrame matches:", self.df[self.df['video_name'] == video_name])
            raise

    def load_flows(self, flows_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flows_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            flow = torch.from_numpy(flow)
            flows.append(flow)
        flows = torch.stack(flows)

        if self.resize:
            flows = F.interpolate(flows, size=self.resize, mode='bilinear')

        return flows.flatten(0, 1)