Instruction of how to implement the code:

I attached a file named "CSYE_7374_Environment.yml" that can be used to replicate my Conda environment. The model is developed using Python 3.9.

Users can replicate the environment by running the following command in Conda:

"conda env create -f CSYE_7374_Environment.yml"

Once the environment is created, you can activate it with:

"conda activate <environment_name>"

I subdivided my code into different sections. Here are the parts that users can modify based on their local device:


---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Section 2:
If you fail to download the dataset, please visit the website:https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
Read the instructions on the website and download the dataset. Deploy the dataset in the "Data" folder.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Section 8:
if save:
      # Save the images during training process
      # You can alter the path to save the images in your local machine
      # Down here is the configuration on my local machine
      # -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
      plt.savefig(f'E:/NEU_CSYE_7374/Final_Project/Visualizations/Diffusion_WithAttention/WithAttention_samples_epoch{epoch}_step{step}.png')
      # -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
      plt.close()
  else:
      plt.show()
      
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Section 9:
# You can modify the hyperparameters here to train the model.
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
self.in_channels = 2  # Image + condition channel
        self.time_channels = 256
        self.n_channels = 64
        self.n_timesteps = 300
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Training parameters
        self.batch_size = 1
    
        self.num_epochs = 80
        self.lr = 1e-4
        self.save_interval = 100  # Save checkpoints every N steps
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #


# Save the checkpoints, you will need to alter the path to save the checkpoints in your local machine
# In the text validation part, you will need to load the checkpoints from the path you save in order to validate the model
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
self.checkpoint_dir = Path("E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention")
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #


# Save the visualizations and log, you will need to alter the path to save the visualizations in your local machine
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
self.vis_dir = Path("E:/NEU_CSYE_7374/Final_Project/Visualizations/Diffusion_WithAttention") 
self.log_dir = Path("E:/NEU_CSYE_7374/Final_Project/Logs/Diffusion_WithAttention")
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Section 10:
# The path to the final checkpoint file of the model:
# You need to alter the path to load the checkpoint in your local machine
# According to which epoch and step you want to load the checkpoint
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch99_step99900.pt"
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #


# Store the generated images in specified directory
# You need to alter the path to save the images in your local machine
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
visualize_validation_samples(generated_samples, save_path="E:/NEU_CSYE_7374/Final_Project/Training_Set_6_Des & Des/Visualizations")
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
# visualize_validation_samples(generated_samples)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Section 11:
# Load the model checkpoint
# You need to alter the path to load the checkpoint in your local machine
# According to which epoch and step you want to load the checkpoint
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
# checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch19_step19900.pt"
# checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch39_step39900.pt"
# checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch59_step59900.pt"
# checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch79_step79900.pt"
checkpoint_with_attention = "E:/NEU_CSYE_7374/Final_Project/Checkpoints/Diffusion_WithAttention/model_epoch99_step99900.pt"
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #


# Input the description text you want to generate the image from model you just loaded
# You need to alter the description text to generate the image you want
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #
description_text = "Lateral chest X-ray showing cardiomegaly with pulmonary edema."
# -------------------------------------------------------------------- Make Changes Here -------------------------------------------------------------------- #

