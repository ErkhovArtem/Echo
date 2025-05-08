from pathlib import Path
import numpy as np


class DataCollector:
    
    """Manages structured dataset collection for robotic manipulation tasks.
    
    Handles:
    - Episode-based data organization
    - Synchronized multi-modal data storage:
      * Robot joint states
      * End-effector poses
      * Gripper positions
      * Force measurements
      * Visual observations (RGB/D)
    - Automatic file management
    
    Directory Structure:
    dataset/
    ├── episode_1.npy
    ├── episode_2.npy
    └── ...
    """

    def __init__(self, dataset_dir="dataset", binary_gripper_pose = False, gripper_config = None):

        """Initialize dataset collection session.
        
        Args:
            dataset_dir (str|Path): Directory to store collected episodes.
                                   Created if doesn't exist.
        """

        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        self.binary_gripper_pose = binary_gripper_pose
        self.gripper_config = gripper_config
        if binary_gripper_pose and gripper_config is None:
            raise TypeError("Gripper config can not be None when binary_gripper_pose=True")
        self.episode = None
        self.task = None


    def add_sample(
        self,
        left_arm,
        right_arm,
        left_arm_new_position,
        right_arm_new_position,
        left_gripper_position,
        right_gripper_position,
        main_camera,
        force,
        wrist_camera = None,
        
    ):
        """Add a synchronized multi-modal data sample.
        
        Args:
            left_arm (UR3Teleop|None): Left arm controller interface
            right_arm (UR3Teleop|None): Right arm controller interface
            left_arm_new_position (np.ndarray[6]): Target joint angles (rad) for left arm
            right_arm_new_position (np.ndarray[6]): Target joint angles (rad) for right arm
            left_gripper_position (int): Target left gripper position [0-255]
            right_gripper_position (int): Target right gripper position [0-255]
            main_camera (RealSenseCamera): Scene-view camera interface
            force (int): Raw force sensor reading [0-4095]
            wrist_camera (WebCamera|None): Optional wrist-mounted camera
            
        Raises:
            RuntimeError: If task description is not set
        """

        if self.episode is None:
            self.episode = []

        step = {}
        step['state'] = np.empty(0, dtype = np.float32)
        step['action'] = np.empty(0, dtype = np.float32)
        for arm, new_position, gripper_position in zip([left_arm, right_arm], [left_arm_new_position, right_arm_new_position], 
                                                       [left_gripper_position, right_gripper_position]): 
            if arm is not None:
                gripper_state =  (self._binarize_gripper_pose(arm.get_current_gripper_pose()) 
                                  if self.binary_gripper_pose 
                                  else arm.get_current_gripper_pose()/255)
                
                step['state'] = np.concat([step['state'],arm.get_current_joint_angles(), arm.get_current_tcp_pose(), 
                                           gripper_state, force/4095]).astype(np.float32)
                
                gripper_action = (self._binarize_gripper_action(gripper_position.item()) 
                                  if self.binary_gripper_pose 
                                  else (np.array([gripper_position.item()]) - arm.get_current_gripper_pose())/255)
                
                step['action'] = np.concat([step['action'], np.array(new_position.tolist()) - arm.get_current_joint_angles(), 
                                            gripper_action]).astype(np.float32)
    
        step['image_main'], step['image_depth'] = main_camera.get_frame()

        if wrist_camera is not None:
            step['image_wrist'] = wrist_camera.get_frame()

        if self.task is not None:
            step['language_instruction'] = self.task
        else:
            raise RuntimeError("Task can not be None.")
        
        
        self.episode.append(step)

    def save_episode(self):
        
        """Save current episode to disk with auto-incremented filename.
        
        File Naming Convention:
        episode_<number>.npy where <number> is the next available integer
        
        Resets episode buffer after saving.
        """

        episode_number = 1
        while (self.dataset_dir / f"episode_{episode_number}.npy").exists():
            episode_number += 1
        np.save(str(self.dataset_dir.absolute()) + f'/episode_{episode_number}.npy', self.episode, allow_pickle=True)
        self.episode = None

    def set_task(self, task):

        """Set the language instruction for subsequent samples.
        
        Args:
            task (str): Natural language description of the current task
                       (e.g., "Pick up the tomato")
        """

        self.task = task
    
    def _binarize_gripper_pose(self, gripper_pose):

        return np.array(gripper_pose > (self.gripper_config["gripper_closed_pose"] + self.gripper_config["gripper_opened_pose"])/2)
    
    def _binarize_gripper_action(self, gripper_action):

        return np.array([gripper_action > self.gripper_config["gripper_pose_threshold"]])


