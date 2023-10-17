import csv
import torch
import os

# Utilities for saving and loading checkpoints, as well as recording to csv
def save_checkpoint(policy_old, episode, save_directory):
    filename = os.path.join(save_directory + "/checkpoints", 'checkpoint_{}.pth'.format(episode))
    torch.save(policy_old.state_dict(), f=filename)
    print('Checkpoint saved to \'{}\''.format(filename))

def load_checkpoint(saves_directory, agent, start):
    if os.path.exists(saves_directory + "/checkpoints"):   
        saved_files = os.listdir(saves_directory + "/checkpoints")
        episode_number = start
        checkpoint_files = [filename for filename in saved_files if filename.startswith("checkpoint_") and filename.endswith(".pth")]
        if checkpoint_files:
            if start == 0:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                print(latest_checkpoint)
                episode_number = int(latest_checkpoint.split('_')[1].split('.')[0])
                agent.episode = episode_number
                agent.model.load_state_dict(torch.load(os.path.join(saves_directory, "checkpoints", latest_checkpoint)))
                agent.model_old.load_state_dict(torch.load(os.path.join(saves_directory, "checkpoints", latest_checkpoint)))
            else:
                agent.episode = start
                checkpoint = "checkpoint_{}.pth".format(start)
                if checkpoint in checkpoint_files:
                    agent.model.load_state_dict(torch.load(os.path.join(saves_directory, "checkpoints", checkpoint)))
                    agent.model_old.load_state_dict(torch.load(os.path.join(saves_directory, "checkpoints", checkpoint)))
                else:
                    print(f"Checkpoint {checkpoint} not found. Starting from episode {start}.")

            print('Resuming training from checkpoint \'{}\'.'.format(episode_number))
        else:
            print("No checkpoint files found.")

def write_to_csv(save_dir, filename, episode, reward):
    with open(os.path.join(save_dir, filename), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward])