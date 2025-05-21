import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Import required R packages
arrow = importr('arrow')
dplyr = importr('dplyr')

# Define the R script path
r_script_path = "/project/aaz/leo/read_parquet.R"

# Execute the R script
robjects.r['source'](r_script_path)

# Or uncomment to run specific parts of your R script:
# robjects.r('''
# folder_path <- "dataset/ceo_scripts_export_20250405_193052"
# result <- read_parquet_data(folder_path)
# if (!is.null(result$dataframe)) {
#   extract_speaker_text(result$dataframe, 100)
# }
# ''')