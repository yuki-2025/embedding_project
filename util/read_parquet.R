# R script to read parquet files and process them
# Similar functionality to read_parquet.py

library(arrow)  # For reading parquet files
library(dplyr)  # For data manipulation

#' Reads all parquet files in a folder in lexicographical order and returns a combined dataframe
#'
#' @param folder_path A string path to the folder containing the parquet files
#' @return A list containing the combined dataframe and the total number of rows
read_parquet_data <- function(folder_path) {
  # Get a list of all parquet files in the folder, sorted lexicographically
  parquet_files <- sort(list.files(folder_path, pattern = "*.parquet", full.names = TRUE))
  
  total_rows <- 0
  dataframes <- list()  # Store dataframes to combine later
  
  for (file_path in parquet_files) {
    tryCatch({
      df <- read_parquet(file_path)
      num_rows <- nrow(df)
      total_rows <- total_rows + num_rows
      dataframes[[length(dataframes) + 1]] <- df  # append df to the dataframes list
      cat(sprintf("File: %s, Rows: %d\n", file_path, num_rows))  # Print rows of individual file
    }, error = function(e) {
      cat(sprintf("Error reading file %s: %s\n", file_path, e$message))
    })
  }
  
  cat(sprintf("\nTotal rows in all files: %d\n", total_rows))
  
  # Combine all dataframes into a single dataframe
  if (length(dataframes) > 0) {
    combined_df <- bind_rows(dataframes)
    return(list(dataframe = combined_df, total_rows = total_rows))
  } else {
    cat("No valid dataframes to process\n")
    return(list(dataframe = NULL, total_rows = 0))
  }
}

#' Gets the first N rows from a dataframe
#'
#' @param dataframe A dataframe
#' @param row Number of rows to return (default: 10000)
#' @return Dataframe with first N rows
get_first_rows <- function(dataframe, row = 10000) {
  # Check if dataframe is empty
  if (is.null(dataframe) || nrow(dataframe) == 0) {
    cat("No data to process\n")
    return(NULL)
  }
  
  tryCatch({
    # Get first N rows
    first_rows <- head(dataframe, row)
    
    cat(sprintf("Retrieved %d rows from the dataframe\n", nrow(first_rows)))
    return(first_rows)
  }, error = function(e) {
    cat(sprintf("Error processing dataframe: %s\n", e$message))
    return(NULL)
  })
}

#' Extracts data from all columns in a dataframe and writes the first num_rows to a text file
#'
#' @param dataframe A dataframe
#' @param num_rows Number of rows to extract and write (default: 10)
#' @return TRUE if successful, FALSE otherwise
extract_speaker_text <- function(dataframe, num_rows = 10) {
  output_file <- "/project/aaz/leo/speaker_text.txt"
  
  # Check if dataframe is empty
  if (is.null(dataframe) || nrow(dataframe) == 0) {
    cat("No data to process\n")
    return(FALSE)
  }
  
  # Take only the first num_rows
  tryCatch({
    if (nrow(dataframe) > num_rows) {
      sample_df <- head(dataframe, num_rows)
    } else {
      sample_df <- dataframe
    }
    
    # Print a preview
    cat("\nPreview of first 5 rows:\n")
    print(head(sample_df, 5))
    
    # Write to file with good formatting
    con <- file(output_file, "w", encoding = "UTF-8")
    
    # Write header
    writeLines(paste0(rep("=", 80), collapse = ""), con)
    writeLines("EXTRACTED DATA", con)
    writeLines(paste0(rep("=", 80), collapse = ""), con)
    writeLines("", con)
    
    # Write each row with all columns
    for (i in 1:nrow(sample_df)) {
      writeLines(sprintf("--- Record %d ---", i), con)
      for (col in names(sample_df)) {
        writeLines(sprintf("%s: %s", col, as.character(sample_df[i, col])), con)
      }
      writeLines("", con)
    }
    
    close(con)
    cat(sprintf("\nFirst %d rows with all columns written to %s\n", nrow(sample_df), output_file))
    return(TRUE)
  }, error = function(e) {
    cat(sprintf("Error processing dataframe: %s\n", e$message))
    return(FALSE)
  })
}

# Example usage:
folder_path <- "dataset/ceo_scripts_export_20250405_193052"  # Replace with actual path
# result <- read_parquet_data(folder_path)
# if (!is.null(result$dataframe)) {
#   extract_speaker_text(result$dataframe, 100)  # Extract 100 rows
# }